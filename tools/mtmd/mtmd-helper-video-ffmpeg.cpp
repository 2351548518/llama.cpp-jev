// mtmd_helper_video_decoder backed by ffmpeg/ffprobe subprocesses

#ifdef MTMD_VIDEO

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd-helper.h"
#include "mtmd-helper-common.h"

#include "hash/hash.h"
#include "sheredom/subprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <fcntl.h>
#include <pthread.h>
#endif

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

namespace {

// RAII wrapper for managing subprocess
struct subprocess_handle {
    struct subprocess_s proc = {};
    bool created = false; // process exists and must be cleaned up
    bool alive   = false; // process can still give us data
    std::thread feeder;

    subprocess_handle() = default;
    subprocess_handle(const subprocess_handle &) = delete;
    subprocess_handle & operator=(const subprocess_handle &) = delete;
    ~subprocess_handle() { stop(); }

    void stop() {
        // note: alive becomes false on stdout EOF, but the process still needs cleanup
        if (!created) {
            return;
        }
        subprocess_terminate(&proc);
#ifdef _WIN32
        // no SIGPIPE on windows: a blocked feeder only gets a broken pipe once we close our read end of the child stdin
        if (proc.hStdInput) {
            CloseHandle(proc.hStdInput);
            proc.hStdInput = nullptr;
        }
#endif
        // join before destroy: feeder holds a FILE* from subprocess_stdin;
        // subprocess_destroy closes it, so the thread must finish first
        if (feeder.joinable()) {
            feeder.join();
        }
        subprocess_join(&proc, nullptr); // reap the child, or else it stays a zombie
        subprocess_destroy(&proc);
        created = false;
        alive   = false;
    }

    FILE * stdout_pipe() {
        return subprocess_stdout(&proc);
    }

    // buf must outlive the feeder thread, i.e. this handle
    void start_feeder(const std::vector<uint8_t> & buf) {
        feeder = std::thread([this, &buf]() {
#ifndef _WIN32
            // ffmpeg can exit before it reads all the input, for example when ffprobe already got the metadata.
            // the write below must then fail with EPIPE, instead of killing the process with SIGPIPE
            sigset_t sigpipe_set;
            sigemptyset(&sigpipe_set);
            sigaddset(&sigpipe_set, SIGPIPE);
            pthread_sigmask(SIG_BLOCK, &sigpipe_set, nullptr); // linux sends the signal to the writing thread
#endif
            FILE * f = subprocess_stdin(&proc);
            if (!f) {
                return;
            }
#ifdef F_SETNOSIGPIPE
            fcntl(fileno(f), F_SETNOSIGPIPE, 1); // macos/bsd send it to the process, so turn it off per fd
#endif
            fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
            proc.stdin_file = nullptr; // prevent double-close in subprocess_destroy
        });
    }
};

std::string resolve_bin(const char * bin_dir, const char * name) {
    if (!bin_dir || bin_dir[0] == '\0') {
        return name; // rely on PATH
    }
    std::string result = bin_dir;
    char last = result.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        result += '\\';
#else
        result += '/';
#endif
    }
    result += name;
#ifdef _WIN32
    result += ".exe";
#endif
    return result;
}

float parse_rational(const char * s) {
    int num = 0, den = 1;
    if (sscanf(s, "%d/%d", &num, &den) == 2 && den > 0) {
        return (float)num / (float)den;
    }
    float val;
    if (sscanf(s, "%f", &val) == 1) {
        return val;
    }
    return 0.0f;
}

double parse_duration_tag(const char * value) {
    double hours, minutes, seconds;
    char suffix;
    if (sscanf(value, "%lf:%lf:%lf%c", &hours, &minutes, &seconds, &suffix) != 3 ||
            !std::isfinite(hours) || !std::isfinite(minutes) || !std::isfinite(seconds) ||
            hours < 0 || hours > INT32_MAX / 3600 || hours != std::floor(hours) ||
            minutes < 0 || minutes >= 60 || minutes != std::floor(minutes) || seconds < 0 || seconds >= 60) {
        return -1.0;
    }
    return hours * 3600 + minutes * 60 + seconds;
}

struct video_probe_cache_entry {
    std::array<uint8_t, 32> hash = {};
    size_t length = 0;
    std::string ffprobe_bin;
    mtmd_helper_video_stream_info info = {};
    uint32_t src_width = 0;
    uint32_t src_height = 0;
    double video_start_s = 0.0;
    bool fixed_rate = false;
};

struct ffmpeg_video {
    std::string path;
    std::vector<uint8_t> input_buf; // non-empty when opened from a buffer
    std::string ffmpeg_bin;
    std::string ffprobe_bin;

    uint32_t src_width     = 0;
    uint32_t src_height    = 0;
    double   video_start_s = 0.0;
    bool     fixed_rate    = false;

    std::vector<float>   pcm;
    std::vector<uint8_t> skip_buf;

    // must be destroyed before input_buf, which its feeder thread reads
    subprocess_handle sp;

    bool is_buf_input() const {
        return !input_buf.empty();
    }

    bool probe(mtmd_helper_video_stream_info & info) {
        // Header validation and decoding probe the same buffer on the same request thread.
        // Keep only metadata, never media bytes; paths bypass the cache because files can change.
        static thread_local std::array<video_probe_cache_entry, 4> cache;
        static thread_local size_t next_entry = 0;
        std::array<uint8_t, 32> hash = {};
        if (is_buf_input()) {
            hash = hash_sha256(input_buf.data(), input_buf.size());
            for (const auto & entry : cache) {
                if (entry.length == input_buf.size() && entry.hash == hash && entry.ffprobe_bin == ffprobe_bin) {
                    info = entry.info;
                    src_width = entry.src_width;
                    src_height = entry.src_height;
                    video_start_s = entry.video_start_s;
                    fixed_rate = entry.fixed_rate;
                    return true;
                }
            }
        }
        // cache: makes stdin seekable; without it ffprobe reports width/height 0
        // for mp4 files whose moov atom sits at the end
        const char * input_arg = is_buf_input() ? "cache:pipe:0" : path.c_str();
        std::vector<const char *> cmd = {
            ffprobe_bin.c_str(),
            "-v", "quiet",
            "-show_entries", "stream=width,height,sample_aspect_ratio,r_frame_rate,avg_frame_rate,nb_frames,duration,start_time:side_data=rotation:stream_tags=rotate,DURATION:format=format_name,duration",
            "-select_streams", "v:0",
            "-of", "default=noprint_wrappers=0",
        };
        if (is_buf_input()) {
            // remove the 64KB read-ahead limit of cache:, or else ffprobe cannot reach a moov atom at end of file
            cmd.insert(cmd.end(), {"-read_ahead_limit", "-1"});
        }
        cmd.insert(cmd.end(), {input_arg, nullptr});

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) { LOG_DBG(" %s", cmd[i]); }
        LOG_DBG("\n");

        subprocess_handle probe_sp;
        if (subprocess_create(cmd.data(),
                subprocess_option_search_user_path | subprocess_option_inherit_environment,
                &probe_sp.proc) != 0) {
            LOG_ERR("%s: failed to launch ffprobe\n", __func__);
            return false;
        }
        probe_sp.created = true;
        probe_sp.alive   = true;

        if (is_buf_input()) {
            probe_sp.start_feeder(input_buf);
        }

        int sar_num  = 1;
        int sar_den  = 1;
        int rotation = 0;
        float avg_fps = 0.0f;
        double tagged_end_s = -1.0;
        double format_duration = -1.0;
        bool in_format = false;
        bool is_matroska = false;
        char line[256];
        FILE * fp = probe_sp.stdout_pipe();

        while (fgets(line, sizeof(line), fp)) {
            line[strcspn(line, "\r\n")] = '\0';
            if (strcmp(line, "[FORMAT]") == 0) {
                in_format = true;
                continue;
            }
            if (strcmp(line, "[/FORMAT]") == 0) {
                in_format = false;
                continue;
            }
            char * eq = strchr(line, '=');
            if (!eq) continue;
            *eq = '\0';
            const char * key = line;
            const char * val = eq + 1;
            char * nl = (char *)strchr(val, '\n');
            if (nl) *nl = '\0';

            if (in_format) {
                if (strcmp(key, "duration") == 0 && strcmp(val, "N/A") != 0) {
                    format_duration = std::strtod(val, nullptr);
                } else if (strcmp(key, "format_name") == 0) {
                    is_matroska = strstr(val, "matroska") != nullptr || strstr(val, "webm") != nullptr;
                }
            } else if (strcmp(key, "width") == 0) {
                info.width = (uint32_t)atoi(val);
            } else if (strcmp(key, "height") == 0) {
                info.height = (uint32_t)atoi(val);
            } else if (strcmp(key, "r_frame_rate") == 0) {
                info.fps = parse_rational(val);
            } else if (strcmp(key, "avg_frame_rate") == 0) {
                avg_fps = parse_rational(val);
            } else if (strcmp(key, "nb_frames") == 0 && strcmp(val, "N/A") != 0) {
                info.n_frames = atoi(val);
            } else if (strcmp(key, "duration") == 0 && strcmp(val, "N/A") != 0) {
                info.duration = (float)atof(val);
            } else if (strcmp(key, "start_time") == 0 && strcmp(val, "N/A") != 0) {
                video_start_s = atof(val);
            } else if (strcmp(key, "sample_aspect_ratio") == 0) {
                if (sscanf(val, "%d:%d", &sar_num, &sar_den) != 2) {
                    sar_num = sar_den = 1; // N/A
                }
            } else if (strcmp(key, "rotation") == 0 || strcmp(key, "TAG:rotate") == 0) {
                rotation = atoi(val);
            } else if (strcmp(key, "TAG:DURATION") == 0 || strcmp(key, "TAG:duration") == 0) {
                tagged_end_s = parse_duration_tag(val);
            }
        }

        probe_sp.stop();

        if (!(std::isfinite(info.duration) && info.duration > 0.0f)) {
            // Matroska's DURATION tag is an end timestamp, including a nonzero stream start.
            const double tagged_duration = tagged_end_s - video_start_s;
            const double fallback = tagged_end_s >= 0.0 && tagged_duration > 0.0 ? tagged_duration :
                format_duration - (is_matroska ? video_start_s : 0.0);
            if (std::isfinite(fallback) && fallback > 0.0 && fallback <= INT32_MAX) {
                info.duration = (float) fallback;
            }
        }

        // Rawvideo output can duplicate frames during rate conversion. Select by source
        // index only when the probe reports a fixed cadence starting at zero.
        fixed_rate = info.fps > 0.0f && info.fps == avg_fps && video_start_s == 0.0 && info.n_frames > 0 &&
            std::abs((double) info.duration * info.fps - info.n_frames) < 0.5;

        // ffmpeg gives frames of the encoded size; the displayed size also has the pixel aspect ratio applied
        src_width  = info.width;
        src_height = info.height;
        const double sar = sar_den > 0 ? (double) sar_num / sar_den : 1.0;
        if (sar >= 0.1 && sar <= 10.0) { // the value is from the file, ignore it if it is not plausible
            info.width = (uint32_t) std::nearbyint(info.width * sar);
        }
        // ffmpeg rotates the frames by default
        if (rotation % 180 == 90 || rotation % 180 == -90) {
            std::swap(src_width, src_height);
            std::swap(info.width, info.height);
        }
        const bool valid = info.width > 0 && info.height > 0 && std::isfinite(info.fps) && info.fps > 0.0f;
        if (valid && is_buf_input()) {
            auto & entry = cache[next_entry++ % cache.size()];
            entry = {hash, input_buf.size(), ffprobe_bin, info, src_width, src_height, video_start_s, fixed_rate};
        }
        return valid;
    }

    void load_soundtrack(int rate, size_t max_samples = SIZE_MAX) {
        const double offset = std::nearbyint(video_start_s * rate);
        if (!std::isfinite(offset) || offset >= (double) SIZE_MAX) {
            return;
        }
        const std::string rate_str = std::to_string(rate);
        std::vector<const char *> cmd = {ffmpeg_bin.c_str(), "-nostdin"};
        cmd.reserve(20); // GCC 11 -Wstringop-overflow false positive on the inserts below otherwise
        if (is_buf_input()) {
            cmd.insert(cmd.end(), {"-read_ahead_limit", "-1"});
        }
        cmd.insert(cmd.end(), {"-i", is_buf_input() ? "cache:pipe:0" : path.c_str(), "-map", "0:a:0", "-vn", "-ac", "1",
                               "-ar", rate_str.c_str(), "-f", "f32le", "pipe:1", "-loglevel", "error", nullptr});
        subprocess_handle audio_sp;
        if (subprocess_create(cmd.data(), subprocess_option_search_user_path | subprocess_option_inherit_environment,
                              &audio_sp.proc) != 0) {
            LOG_WRN("%s: failed to launch ffmpeg for the soundtrack\n", __func__);
            return;
        }
        audio_sp.created = true;
        audio_sp.alive   = true;
        if (is_buf_input()) {
            audio_sp.start_feeder(input_buf);
        }
        float block[4096];
        FILE * out = audio_sp.stdout_pipe();
        size_t skip = (size_t) std::max(0.0, offset);
        while (skip > 0) {
            const size_t n = fread(block, sizeof(float), std::min(skip, (size_t) 4096), out);
            if (n == 0) {
                return;
            }
            skip -= n;
        }
        size_t n;
        while (pcm.size() < max_samples && (n = fread(block, sizeof(float), std::min(max_samples - pcm.size(), (size_t) 4096), out)) > 0) {
            pcm.insert(pcm.end(), block, block + n);
        }
    }

    bool start(uint32_t width, uint32_t height, float fps, std::string vf = {}, size_t n_selected = 0) {
        std::vector<const char *> cmd;
        cmd.push_back(ffmpeg_bin.c_str());

        cmd.push_back("-nostdin");
        if (is_buf_input()) {
            // remove the 64KB read-ahead limit of cache:, or else ffmpeg cannot reach a moov atom at end of file
            cmd.push_back("-read_ahead_limit");
            cmd.push_back("-1");
        }
        cmd.push_back("-i");
        // cache:pipe:0 wraps stdin with a seekable in-memory cache, letting ffmpeg seek
        // backwards for container headers (e.g. MP4 moov atom at end of file)
        cmd.push_back(is_buf_input() ? "cache:pipe:0" : path.c_str());

        const bool scale = width != src_width || height != src_height;
        char filter[160] = "";
        if (fps > 0.0f) {
            if (scale) {
                snprintf(filter, sizeof(filter), "fps=%.6f,scale=%u:%u", fps, width, height);
            } else {
                snprintf(filter, sizeof(filter), "fps=%.6f", fps);
            }
        } else if (scale) {
            // B=0 C=0.5 bicubic on RGB is closest to Pillow
            snprintf(filter, sizeof(filter),
                    "format=rgb24,scale=%u:%u:flags=bicubic+accurate_rnd+full_chroma_int:param0=0:param1=0.5",
                    width, height);
        }
        if (filter[0]) {
            if (!vf.empty()) {
                vf += ',';
            }
            vf += filter;
        }
        if (!vf.empty()) {
            cmd.insert(cmd.end(), {"-vf", vf.c_str()});
        }
        const std::string count = std::to_string(n_selected);
        if (n_selected > 0) {
            cmd.insert(cmd.end(), {"-vsync", "0", "-frames:v", count.c_str()});
        }

        cmd.push_back("-f");
        cmd.push_back("rawvideo");
        cmd.push_back("-pix_fmt");
        cmd.push_back("rgb24");
        cmd.push_back("pipe:1");
        cmd.push_back("-loglevel");
        cmd.push_back("error");
        cmd.push_back(nullptr);

        LOG_DBG("%s: launching:", __func__);
        for (size_t i = 0; cmd[i]; i++) {
            LOG_DBG(" %s", cmd[i]);
        }
        LOG_DBG("\n");

        int ret = subprocess_create(
            cmd.data(),
            subprocess_option_search_user_path | subprocess_option_inherit_environment,
            &sp.proc);

        sp.created = (ret == 0);
        sp.alive   = (ret == 0);
        LOG_DBG("%s: subprocess_create ret=%d proc_alive=%d\n", __func__, ret, (int)sp.alive);
        if (!sp.alive) {
            LOG_ERR("%s: failed to start ffmpeg (is ffmpeg in PATH?)\n", __func__);
            return false;
        }

        if (is_buf_input()) {
            LOG_DBG("%s: starting feeder thread for %zu-byte buffer\n", __func__, input_buf.size());
            sp.start_feeder(input_buf);
        }
        return true;
    }

    bool read(unsigned char * dst, size_t n) {
        FILE * fp = sp.stdout_pipe();
        if (!dst) {
            skip_buf.resize(n);
            dst = skip_buf.data();
        }
        size_t total_read = 0;
        while (total_read < n) {
            const size_t got = fread(dst + total_read, 1, n - total_read, fp);
            if (got == 0) {
                // clean EOF only if no bytes read yet; partial frame is an error
                LOG_DBG("%s: fread returned 0 after %zu/%zu bytes (ferror=%d)\n",
                        __func__, total_read, n, ferror(fp));
                return false;
            }
            total_read += got;
        }
        return true;
    }
};

void * ffmpeg_open(void * user_data, const char * path, const unsigned char * buf, size_t len,
                   mtmd_helper_video_stream_info * info) {
    auto v = std::make_unique<ffmpeg_video>();
    const char * bin_dir = static_cast<const char *>(user_data);
    v->ffmpeg_bin  = resolve_bin(bin_dir, "ffmpeg");
    v->ffprobe_bin = resolve_bin(bin_dir, "ffprobe");
    if (buf && len > 0) {
        v->input_buf.assign(buf, buf + len);
    } else if (path) {
        v->path = path;
    }
    if (!v->probe(*info)) {
        LOG_ERR("%s: ffprobe failed for '%s' (is ffprobe in PATH?)\n", __func__, v->is_buf_input() ? "<buffer>" : v->path.c_str());
        return nullptr;
    }
    return v.release();
}

bool ffmpeg_start(void * state, uint32_t width, uint32_t height, float fps, bool * fps_picked) {
    *fps_picked = fps > 0.0f;
    return static_cast<ffmpeg_video *>(state)->start(width, height, fps);
}

bool ffmpeg_start_selected(void * state, uint32_t width, uint32_t height, const int32_t * indices, size_t n_indices) {
    auto * video = static_cast<ffmpeg_video *>(state);
    if (!video->fixed_rate) {
        return false;
    }

    std::string filter = "select=";
    int32_t previous = -1;
    size_t count = 0;
    for (size_t i = 0; i < n_indices; i++) {
        if (indices[i] == previous) {
            continue;
        }
        if (count++ > 0) {
            filter += '+';
        }
        filter += "eq(n\\," + std::to_string(indices[i]) + ")";
        previous = indices[i];
    }
    return video->start(width, height, 0.0f, std::move(filter), count);
}

bool ffmpeg_read_frame(void * state, unsigned char * rgb, uint32_t width, uint32_t height) {
    auto * v = static_cast<ffmpeg_video *>(state);
    if (!v->sp.alive) {
        return false;
    }
    if (!v->pcm.empty()) {
        std::vector<float>().swap(v->pcm); // no longer needed
    }
    LOG_DBG("%s: reading a frame, expecting %zu bytes (%ux%u)\n", __func__, (size_t) width * height * 3, width, height);
    return v->read(rgb, (size_t) width * height * 3);
}

const float * ffmpeg_read_audio_bounded(void * state, int sample_rate, size_t max_samples, size_t * n_samples) {
    auto * v = static_cast<ffmpeg_video *>(state);
    v->pcm.clear();
    v->load_soundtrack(sample_rate, max_samples);
    *n_samples = v->pcm.size();
    return *n_samples > 0 ? v->pcm.data() : nullptr;
}

const float * ffmpeg_read_audio(void * state, int sample_rate, size_t * n_samples) {
    return ffmpeg_read_audio_bounded(state, sample_rate, SIZE_MAX, n_samples);
}

void ffmpeg_close(void * state) {
    delete static_cast<ffmpeg_video *>(state);
}

} // namespace

const mtmd_helper_video_decoder * mtmd_helper_video_decoder_ffmpeg() {
    static const mtmd_helper_video_decoder decoder = {
        /* open       */ ffmpeg_open,
        /* start      */ ffmpeg_start,
        /* read_frame */ ffmpeg_read_frame,
        /* read_audio */ ffmpeg_read_audio,
        /* close      */ ffmpeg_close,
        /* user_data  */ nullptr,
        /* read_audio_bounded */ ffmpeg_read_audio_bounded,
        /* start_selected */ ffmpeg_start_selected,
    };
    return &decoder;
}

#endif // MTMD_VIDEO
