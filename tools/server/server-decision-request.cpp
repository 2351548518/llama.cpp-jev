#include "server-decision.h"

#include "base64.hpp"
#include "server-http.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>

static constexpr size_t MAX_ASSET_REQUEST_BYTES = 32 * 1024 * 1024;
static constexpr size_t MAX_ASSET_IMAGE_BYTES = 16 * 1024 * 1024;
static constexpr uint32_t MIN_ASSET_SIDE = 280;
static constexpr uint32_t MAX_ASSET_SIDE = 1568;
static constexpr int MAX_ASSET_DEPTH = 128;

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value;
}

static std::string read_header(const server_http_req & req, const std::string & key) {
    for (const auto & [name, value] : req.headers) {
        if (lower_ascii(name) == key) {
            return value;
        }
    }
    return {};
}

static bool images_enabled(const server_http_req & req) {
    return lower_ascii(string_strip(read_header(req, "typesafe-danger-zone-images"))) == "true";
}

static std::string media_type(const std::string & content_type) {
    return lower_ascii(string_strip(content_type.substr(0, content_type.find(';'))));
}

json server_decision_request_json(const server_http_req & req, bool native_media) {
    if (native_media && (req.multipart || req.body.size() > 48 * 1024 * 1024)) {
        throw std::invalid_argument("multimodal requires a JSON body of at most 48 MiB");
    }
    if (!native_media && (req.multipart || images_enabled(req))) {
        size_t bytes = req.body.size();
        if (bytes > MAX_ASSET_REQUEST_BYTES) {
            throw std::invalid_argument("System One image request exceeds 32 MiB");
        }
        for (const auto & [name, file] : req.files) {
            if (file.data.size() > MAX_ASSET_REQUEST_BYTES - bytes) {
                throw std::invalid_argument("System One image request exceeds 32 MiB");
            }
            bytes += file.data.size();
        }
    }
    json body = json::parse(req.body);
    if (!body.is_object()) {
        throw std::invalid_argument("System One request must be a JSON object");
    }
    if (req.multipart) {
        const auto part = req.files.find("request");
        if (body.size() == 1 && body.contains("request") && body.at("request").is_string() && part == req.files.end()) {
            body = json::parse(body.at("request").get<std::string>());
        } else if (body.empty() && part != req.files.end() && media_type(part->second.content_type) == "application/json") {
            body = json::parse(std::string(part->second.data.begin(), part->second.data.end()));
        } else {
            throw std::invalid_argument("multipart requires one JSON part named 'request' and accepts named image uploads");
        }
        if (!body.is_object() || !body.contains("assets") || !body.at("assets").is_object() || body.at("assets").empty()) {
            throw std::invalid_argument("multipart System One requests need image assets");
        }
    }
    return body;
}

bool server_decision_typesafe_response(const server_http_req & req, bool has_assets, bool native_media) {
    const std::string schema = lower_ascii(string_strip(read_header(req, "x-laya-response-schema")));
    if (!schema.empty()) {
        if (schema != "typesafe" && schema != "legacy") {
            throw std::invalid_argument("X-Laya-Response-Schema must be typesafe or legacy");
        }
        return schema == "typesafe";
    }
    const char * setting = std::getenv("LAYA_JEV_STRICT");
    const std::string strict = setting ? lower_ascii(setting) : "";
    return has_assets || (!native_media && (strict == "1" || strict == "true" || strict == "yes" || strict == "on"));
}

static raw_buffer decode_file(const json & value, size_t bytes_limit) {
    if (!value.is_string()) {
        throw std::invalid_argument("expected base64 file bytes");
    }
    const std::string data = value.get<std::string>();
    if (data.empty() || data.size() % 4 != 0) {
        throw std::invalid_argument("invalid base64 file bytes");
    }
    const size_t padding = data.back() == '=' ? (data[data.size() - 2] == '=' ? 2 : 1) : 0;
    const size_t size = data.size() / 4 * 3 - padding;
    if (size > bytes_limit) {
        throw std::invalid_argument("decoded media exceeds the request size limit");
    }
    raw_buffer result;
    result.reserve(size);
    try {
        base64::decode(data.begin(), data.end(), std::back_inserter(result), base64::alphabet::standard);
    } catch (const base64_error &) {
        throw std::invalid_argument("invalid base64 file bytes");
    }
    if (result.size() != size || result.empty()) {
        throw std::invalid_argument("invalid base64 file bytes");
    }
    return result;
}

std::string server_decision_read_media(const json & body, std::vector<raw_buffer> & files) {
    std::string kind;
    for (const auto & [key, value] : body.items()) {
        if (key == "image" || key == "video" || key == "audio") {
            if (value.is_null()) {
                continue;
            }
            if (!kind.empty() || !value.is_object() || value.size() != 1 || !value.contains("base64")) {
                throw std::invalid_argument("supply exactly one image, video or audio object containing base64");
            }
            kind = key;
            files.push_back(decode_file(value.at("base64"), MAX_ASSET_REQUEST_BYTES));
        } else if (key != "state" && key != "questions" && key != "model" && key != "cache_prompt" &&
                   key != "max_len" && key != "min_confidence") {
            throw std::invalid_argument("unsupported multimodal request field: " + key);
        }
    }
    if (kind.empty()) {
        throw std::invalid_argument("supply exactly one image, video or audio object containing base64");
    }
    return kind;
}

void server_decision_validate_confidence(const json & limit) {
    auto check = [](const json & value) {
        if (!value.is_number() || !std::isfinite(value.get<double>()) || value.get<double>() < 0 || value.get<double>() > 1) {
            throw std::invalid_argument("min_confidence thresholds must be numbers in [0, 1]");
        }
    };
    if (limit.is_object()) {
        for (const auto & [key, value] : limit.items()) {
            const auto colon = key.find(':');
            const std::string type = key.substr(0, colon);
            const std::string bucket = colon == std::string::npos ? "" : key.substr(colon + 1);
            if (key != "default" && ((type != "choice" && type != "score" && type != "noul") ||
                    (bucket != "2" && bucket != "3-5" && bucket != "6-10" && bucket != "11+"))) {
                throw std::invalid_argument("min_confidence needs a decision type and option-count bucket");
            }
            check(value);
        }
    } else if (!limit.is_null()) {
        check(limit);
    }
}

void server_decision_apply_confidence(json & answer, const json & limit) {
    if (limit.is_null()) {
        return;
    }
    const auto type = answer.at("type").get<std::string>();
    double threshold = limit.is_object() ? limit.value("default", 0.0) : limit.get<double>();
    if (limit.is_object() && type != "box") {
        const size_t count = type == "noul" ? 2 : answer.at("probabilities").size();
        const auto bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+";
        threshold = limit.value(type + ":" + bucket, threshold);
    }
    double confidence = 0.;
    if (answer.contains("answer_confidence")) {
        confidence = answer.at("answer_confidence").get<double>();
    } else if (type == "noul") {
        const double p = answer.at("noul").get<double>();
        confidence = std::max(p, 1. - p);
    } else if (type == "box") {
        confidence = answer.at("confidence").get<double>();
    } else {
        for (const auto & p : answer.at("probabilities")) {
            confidence = std::max(confidence, p.get<double>());
        }
    }
    answer["abstention_threshold"] = threshold;
    answer["abstention"] = confidence < threshold ? "abstained" : "passed";
    if (confidence < threshold) {
        answer["low_confidence"] = true;
    }
}

static bool is_animated_png(const raw_buffer & data) {
    for (size_t offset = 8; offset + 12 <= data.size();) {
        const size_t size = (uint32_t(data[offset]) << 24) | (uint32_t(data[offset + 1]) << 16) |
                            (uint32_t(data[offset + 2]) << 8) | uint32_t(data[offset + 3]);
        if (size > data.size() - offset - 12) {
            break;
        }
        const std::string_view type(reinterpret_cast<const char *>(data.data() + offset + 4), 4);
        if (type == "acTL") {
            return true;
        }
        if (type == "IDAT") {
            break;
        }
        offset += size + 12;
    }
    return false;
}

static size_t asset_pixels(const raw_buffer & data, const std::string & format, const mtmd_helper_init_opt & options) {
    const std::array<uint8_t, 8> png_signature = { 137, 80, 78, 71, 13, 10, 26, 10 };
    const bool png = data.size() >= png_signature.size() && std::equal(png_signature.begin(), png_signature.end(), data.begin());
    const bool jpeg = data.size() >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff;
    if ((!png && !jpeg) || (format == "png" && !png) || ((format == "jpg" || format == "jpeg") && !jpeg) ||
        (png && is_animated_png(data))) {
        throw std::invalid_argument("assets must contain still PNG or JPEG bytes matching their declared format");
    }
    mtmd_helper_media_meta meta = {};
    if (mtmd_helper_get_media_meta(data.data(), data.size(), options, &meta) != 0 ||
        meta.type != MTMD_HELPER_MEDIA_TYPE_IMAGE || std::min(meta.width, meta.height) < MIN_ASSET_SIDE ||
        std::max(meta.width, meta.height) > MAX_ASSET_SIDE) {
        throw std::invalid_argument("asset image width and height must be between 280 and 1568 pixels");
    }
    return size_t(meta.width) * meta.height * 3;
}

server_decision_assets server_decision_bind_assets(const server_http_req & req, json & body,
        std::vector<raw_buffer> & files, const mtmd_helper_init_opt & options) {
    if (!images_enabled(req)) {
        throw std::invalid_argument("set TypeSafe-Danger-Zone-Images: true for image assets");
    }
    if (body.contains("files") || body.contains("images") || body.contains("audio") || body.contains("video")) {
        throw std::invalid_argument("combine image placeholders through assets only");
    }
    if (!body.contains("state") || !body.contains("questions") || !body.at("questions").is_object()) {
        throw std::invalid_argument("image assets need state and a questions object");
    }

    server_decision_assets assets = { "<__laya_asset_" + random_string() + "_", {} };
    std::vector<std::string> names;
    std::vector<size_t> pixels;
    std::set<std::string> used_parts;
    size_t decoded_bytes = 0;
    size_t name_bytes = 0;
    for (const auto & [name, asset] : body.at("assets").items()) {
        name_bytes += name.size();
        if (string_strip(name).empty() || name.size() > 1024 || name_bytes > 65536 ||
            !asset.is_object() || (asset.contains("type") && asset.at("type") != "image")) {
            throw std::invalid_argument("image assets need nonempty, bounded names and optional type image");
        }
        const auto format = json_value(asset, "format", std::string());
        if (asset.contains("format") && format != "png" && format != "jpg" && format != "jpeg") {
            throw std::invalid_argument("image format must be png, jpg or jpeg");
        }
        if (asset.contains("part")) {
            const auto part = json_value(asset, "part", std::string());
            const auto found = req.files.find(part);
            if (part.empty() || part == "request" || found == req.files.end() || !used_parts.insert(part).second) {
                throw std::invalid_argument("each asset part must reference a distinct uploaded file");
            }
            if (found->second.data.size() > MAX_ASSET_IMAGE_BYTES) {
                throw std::invalid_argument("image file exceeds 16 MiB");
            }
            const auto content_type = media_type(found->second.content_type);
            if (!content_type.empty() && content_type != "image/png" && content_type != "image/jpeg" &&
                content_type != "image/jpg" && content_type != "application/octet-stream") {
                throw std::invalid_argument("image parts need a PNG, JPEG or application/octet-stream content type");
            }
            files.push_back(found->second.data);
        } else {
            if (format.empty() || !asset.contains("data_base64")) {
                throw std::invalid_argument("inline image assets need format and data_base64");
            }
            files.push_back(decode_file(asset.at("data_base64"), MAX_ASSET_IMAGE_BYTES));
        }
        names.push_back(name);
        pixels.push_back(asset_pixels(files.back(), format, options));
        decoded_bytes += pixels.back();
        if (decoded_bytes > MAX_ASSET_REQUEST_BYTES) {
            throw std::invalid_argument("decoded image assets exceed 32 MiB");
        }
    }
    for (const auto & [name, file] : req.files) {
        if (name != "request" && !used_parts.count(name)) {
            throw std::invalid_argument("every uploaded image must be referenced by an asset");
        }
    }

    auto & state = body.at("state");
    if (state.is_array() && state.size() == names.size() + 1 && state.back().is_string()) {
        std::set<size_t> distinct;
        for (size_t i = 0; i < names.size() && state[i].is_string(); ++i) {
            const auto found = std::find(names.begin(), names.end(), state[i].get<std::string>());
            if (found == names.end() || !distinct.insert(size_t(found - names.begin())).second) {
                break;
            }
            assets.prefix_order.push_back(size_t(found - names.begin()));
        }
        if (assets.prefix_order.size() != names.size()) {
            assets.prefix_order.clear();
        }
    }
    const json prefix_state = assets.prefix_order.empty() ? json() : state.back();
    std::vector<size_t> by_length(names.size());
    std::iota(by_length.begin(), by_length.end(), 0);
    std::sort(by_length.begin(), by_length.end(), [&](size_t a, size_t b) { return names[a].size() > names[b].size(); });

    size_t expanded_bytes = 0;
    std::vector<bool> used(names.size());
    auto bind = [&](auto && self, json & value, int depth) -> bool {
        if (depth >= MAX_ASSET_DEPTH) {
            throw std::invalid_argument("image references are nested too deeply");
        }
        if (value.is_array() || value.is_object()) {
            bool changed = false;
            for (auto & item : value) {
                changed |= self(self, item, depth + 1);
            }
            return changed;
        }
        if (!value.is_string()) {
            return false;
        }
        const std::string text = value.get<std::string>();
        if (text.find(get_media_marker()) != std::string::npos || text.find(assets.marker) != std::string::npos) {
            throw std::invalid_argument("text contains a reserved media marker");
        }
        std::string rendered;
        size_t offset = 0;
        while (offset < text.size()) {
            size_t next = std::string::npos;
            size_t index = 0;
            for (size_t i : by_length) {
                const size_t found = text.find(names[i], offset);
                if (found < next) {
                    next = found;
                    index = i;
                }
            }
            if (next == std::string::npos) {
                break;
            }
            expanded_bytes += pixels[index];
            if (expanded_bytes > MAX_ASSET_REQUEST_BYTES) {
                throw std::invalid_argument("expanded image references exceed 32 MiB");
            }
            used[index] = true;
            rendered.append(text, offset, next - offset);
            rendered += assets.marker + std::to_string(index) + ">";
            offset = next + names[index].size();
        }
        if (offset == 0) {
            return false;
        }
        rendered.append(text, offset, std::string::npos);
        value = std::move(rendered);
        return true;
    };
    bind(bind, state, 0);
    bool questions_changed = false;
    for (auto & question : body.at("questions")) {
        if (question.is_object() && json_value(question, "type", std::string()) == "choice" &&
            question.contains("criteria") && !question.at("criteria").is_object()) {
            throw std::invalid_argument("Choice with image assets needs a named criteria object");
        }
        for (const char * field : { "instructions", "criteria" }) {
            if (question.contains(field)) {
                questions_changed |= bind(bind, question.at(field), 0);
            }
        }
    }
    if (std::find(used.begin(), used.end(), false) != used.end()) {
        throw std::invalid_argument("each image asset must be referenced in state, instructions or criteria");
    }
    if (!assets.prefix_order.empty()) {
        if (!questions_changed && state.back() == prefix_state) {
            state = prefix_state;
        } else {
            assets.prefix_order.clear();
        }
    }
    return assets;
}

std::vector<size_t> server_decision_assets::resolve(
        std::string & text, std::vector<std::pair<size_t, size_t>> & spans) const {
    std::vector<size_t> order;
    std::string rendered;
    size_t offset = 0, span = 0;
    auto append_text = [&](size_t end) {
        for (; span < spans.size() && spans[span].second <= end; ++span) {
            auto & [first, last] = spans[span];
            if (first < offset) {
                throw std::runtime_error("media reference overlaps a candidate suffix");
            }
            first = rendered.size() + (first - offset);
            last = rendered.size() + (last - offset);
        }
        rendered.append(text, offset, end - offset);
    };
    for (size_t next = text.find(marker, offset); next != std::string::npos; next = text.find(marker, offset)) {
        append_text(next);
        const size_t start = next + marker.size();
        const size_t end = text.find('>', start);
        if (end == std::string::npos) {
            throw std::runtime_error("invalid bound image reference");
        }
        order.push_back(std::stoul(text.substr(start, end - start)));
        rendered += get_media_marker();
        offset = end + 1;
    }
    if (!order.empty()) {
        append_text(text.size());
        text = std::move(rendered);
    }
    return order;
}
