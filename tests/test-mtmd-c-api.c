#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#undef NDEBUG
#include <assert.h>

#include "mtmd.h"
#include "mtmd-helper.h"

struct native_video_counts {
    int opened;
    int started;
    int read;
    int closed;
};

static void * native_video_open(void * user_data, const char * path, const unsigned char * buf, size_t len,
                                struct mtmd_helper_video_stream_info * info) {
    (void) path;
    if (!buf || len < 12) {
        return NULL;
    }
    struct native_video_counts * counts = (struct native_video_counts *) user_data;
    counts->opened++;
    *info = (struct mtmd_helper_video_stream_info) {2, 2, 2.0f, 4, 2.0f};
    return counts;
}

static bool native_video_start(void * state, uint32_t width, uint32_t height, float fps, bool * fps_picked) {
    (void) fps;
    struct native_video_counts * counts = (struct native_video_counts *) state;
    counts->started++;
    *fps_picked = false;
    return width > 0 && height > 0 && width <= 4096 && height <= 4096;
}

static bool native_video_read(void * state, unsigned char * rgb, uint32_t width, uint32_t height) {
    struct native_video_counts * counts = (struct native_video_counts *) state;
    if (counts->read++ >= 4 || width == 0 || height == 0 || width > 4096 || height > 4096) {
        return false;
    }
    if (rgb) {
        memset(rgb, 17, (size_t) width * height * 3);
    }
    return true;
}

static void native_video_close(void * state) {
    ((struct native_video_counts *) state)->closed++;
}

static void test_native_video_decoder(void) {
    struct native_video_counts counts = {0};
    struct mtmd_helper_video_decoder decoder = {0};
    decoder.open = native_video_open;
    decoder.start = native_video_start;
    decoder.read_frame = native_video_read;
    decoder.close = native_video_close;
    decoder.user_data = &counts;
    struct mtmd_helper_init_opt options = mtmd_helper_init_opt_default();
    options.video_params.decoder = &decoder;
    const unsigned char video[] = "native-video-container";
    struct mtmd_helper_media_meta meta;
    assert(mtmd_helper_get_media_meta(video, sizeof(video), options, &meta) == 0);
    assert(meta.type == MTMD_HELPER_MEDIA_TYPE_VIDEO && meta.width == 2 && meta.height == 2);
    assert(meta.duration == 2.0f && meta.n_frames == 4);
    assert(counts.opened == 1 && counts.closed == 1 && counts.started == 0);

#ifdef MTMD_TEST_NO_FFMPEG
    const unsigned char webp[] = "RIFF\0\0\0\0WEBP";
    struct mtmd_helper_bitmap_wrapper image = mtmd_helper_bitmap_init_from_buf(NULL, webp, sizeof(webp), false, options);
    assert(image.bitmap != NULL && image.video_ctx == NULL);
    assert(mtmd_bitmap_get_nx(image.bitmap) == 2 && mtmd_bitmap_get_ny(image.bitmap) == 2);
    assert(mtmd_bitmap_get_data(image.bitmap)[0] == 17);
    mtmd_bitmap_free(image.bitmap);
    assert(counts.opened == 2 && counts.closed == 2 && counts.started == 1 && counts.read == 1);
#endif

    const char * model_path = getenv("MTMD_TEST_MODEL");
    const char * projector_path = getenv("MTMD_TEST_MMPROJ");
    if (model_path && projector_path) {
        struct llama_model_params model_params = llama_model_default_params();
        model_params.vocab_only = true;
        model_params.n_gpu_layers = 0;
        struct llama_model * model = llama_model_load_from_file(model_path, model_params);
        assert(model != NULL);
        struct mtmd_context_params context_params = mtmd_context_params_default();
        context_params.use_gpu = false;
        context_params.warmup = false;
        context_params.n_threads = 2;
        struct mtmd_context * context = mtmd_init_from_file(projector_path, model, context_params);
        assert(context != NULL);
        memset(&counts, 0, sizeof(counts));
        options.video_params.center_frames = 4;
        options.video_params.prompt_template = MTMD_VIDEO_TEMPLATE_QWEN3VL_WRAPPED;
        options.video_params.no_soundtrack = true;
        mtmd_helper_video * reader = mtmd_helper_video_init_from_buf(context, video, sizeof(video), options.video_params);
        assert(reader != NULL);
        int frames = 0;
        int code;
        mtmd_bitmap * bitmap;
        char * text;
        while ((code = mtmd_helper_video_read_next(reader, &bitmap, &text)) == 0) {
            if (bitmap) {
                assert(mtmd_bitmap_get_data(bitmap)[0] == 17);
                frames++;
                mtmd_bitmap_free(bitmap);
            }
            free(text);
        }
        assert(code == -1 && frames == 4);
        mtmd_helper_video_free(reader);
        assert(counts.opened == 1 && counts.closed == 1 && counts.started == 1);
        mtmd_free(context);
        llama_model_free(model);
        printf("Native callback video: four frames and balanced decoder lifetime\n");
    }

    decoder.read_frame = NULL;
    assert(mtmd_helper_get_media_meta(video, sizeof(video), options, &meta) != 0);
    assert(counts.opened == counts.closed);
}

int main(void) {
    printf("\n\nTesting libmtmd C API...\n");
    printf("--------\n\n");
    test_native_video_decoder();

    struct mtmd_context_params params = mtmd_context_params_default();
    printf("Default image marker: %s\n", params.image_marker);

    mtmd_input_chunks * chunks = mtmd_test_create_input_chunks();

    if (!chunks) {
        fprintf(stderr, "Failed to create input chunks\n");
        return 1;
    }

    // simple test for the helper
    size_t n_tokens_total = mtmd_helper_get_n_tokens(chunks);
    printf("Total tokens in chunks: %zu\n", n_tokens_total);
    assert(n_tokens_total > 0);

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    printf("Number of chunks: %zu\n", n_chunks);
    assert(n_chunks > 0);

    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        assert(chunk != NULL);
        enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        printf("Chunk %zu type: %d\n", i, type);

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            printf("    Text chunk with %zu tokens\n", n_tokens);
            assert(tokens != NULL);
            assert(n_tokens > 0);
            for (size_t j = 0; j < n_tokens; j++) {
                assert(tokens[j] >= 0);
                printf("    > Token %zu: %d\n", j, tokens[j]);
            }

        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            size_t n_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);
            // get position of the last token, which should be (nx - 1, ny - 1)
            struct mtmd_decoder_pos pos = mtmd_image_tokens_get_decoder_pos(image_tokens, 0, n_tokens - 1);
            size_t nx = pos.x + 1;
            size_t ny = pos.y + 1;
            const char * id = mtmd_image_tokens_get_id(image_tokens);
            assert(n_tokens > 0);
            assert(nx > 0);
            assert(ny > 0);
            assert(id != NULL);
            printf("    Image chunk with %zu tokens\n", n_tokens);
            printf("    Image size: %zu x %zu\n", nx, ny);
            printf("    Image ID: %s\n", id);
        }
    }

    // test chunk save/load round-trip
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        assert(chunk != NULL);
        enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);

        // query the required buffer size (out_buf == NULL)
        size_t expected_len = 0;
        int32_t rc = mtmd_input_chunk_save(chunk, NULL, 0, &expected_len);
        printf("    Chunk %zu: save query rc = %d, expected_len = %zu\n", i, rc, expected_len);
        assert(rc == 0);
        assert(expected_len > 0);

        // saving into a too-small buffer must fail, not crash
        char tiny_buf[1];
        rc = mtmd_input_chunk_save(chunk, tiny_buf, sizeof(tiny_buf), NULL);
        printf("    Chunk %zu: save into too-small buffer rc = %d (expect non-zero)\n", i, rc);
        assert(rc != 0);

        // save into a properly-sized buffer
        char * buf = (char *) malloc(expected_len);
        assert(buf != NULL);
        rc = mtmd_input_chunk_save(chunk, buf, expected_len, NULL);
        assert(rc == 0);

        // loading from a truncated buffer must fail gracefully, not crash
        if (expected_len > 1) {
            mtmd_input_chunk * bad = mtmd_input_chunk_load(buf, expected_len - 1);
            printf("    Chunk %zu: load from truncated buffer = %p (expect NULL)\n", i, (void *) bad);
            assert(bad == NULL);
        }

        // load it back
        mtmd_input_chunk * loaded = mtmd_input_chunk_load(buf, expected_len);
        assert(loaded != NULL);

        // metadata must match the original chunk
        assert(mtmd_input_chunk_get_type(loaded) == type);
        assert(mtmd_input_chunk_get_n_tokens(loaded) == mtmd_input_chunk_get_n_tokens(chunk));
        assert(mtmd_input_chunk_get_n_pos(loaded) == mtmd_input_chunk_get_n_pos(chunk));

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tok_orig, n_tok_loaded;
            const llama_token * tok_orig   = mtmd_input_chunk_get_tokens_text(chunk, &n_tok_orig);
            const llama_token * tok_loaded = mtmd_input_chunk_get_tokens_text(loaded, &n_tok_loaded);
            printf("    Chunk %zu: loaded %zu text tokens (orig %zu), first token %d (orig %d)\n",
                i, n_tok_loaded, n_tok_orig,
                n_tok_loaded > 0 ? tok_loaded[0] : -1,
                n_tok_orig   > 0 ? tok_orig[0]   : -1);
            assert(n_tok_orig == n_tok_loaded);
            for (size_t j = 0; j < n_tok_orig; j++) {
                assert(tok_orig[j] == tok_loaded[j]);
            }
        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            const char * id_orig   = mtmd_input_chunk_get_id(chunk);
            const char * id_loaded = mtmd_input_chunk_get_id(loaded);
            printf("    Chunk %zu: loaded id '%s' (orig '%s')\n", i, id_loaded, id_orig);
            assert(id_orig != NULL && id_loaded != NULL);
            assert(strcmp(id_orig, id_loaded) == 0);
        }

        mtmd_input_chunk_free(loaded);
        free(buf);
    }
    printf("Chunk save/load round-trip OK\n");

    // test input validation of mtmd_tokenize_from_parts()
    // invalid parts are rejected before the ctx is used, so NULL ctx is OK here
    {
        mtmd_input_chunks * out = mtmd_input_chunks_init();
        mtmd_bitmap * bmp = mtmd_bitmap_init(4, 4, NULL); // placeholder bitmap
        struct mtmd_input_text txt = { "hello", 5, false, false };
        struct mtmd_input_text txt_null = { NULL, 0, false, false };

        struct mtmd_input_part part_both      = { &txt, bmp };
        struct mtmd_input_part part_neither   = { NULL, NULL };
        struct mtmd_input_part part_null_text = { &txt_null, NULL };
        const mtmd_input_part * parts[1];
        int32_t rc;

        parts[0] = &part_both;
        rc = mtmd_tokenize_from_parts(NULL, out, parts, 1, false);
        printf("tokenize part with both text and bitmap rc = %d (expect 1)\n", rc);
        assert(rc == 1);

        parts[0] = &part_neither;
        rc = mtmd_tokenize_from_parts(NULL, out, parts, 1, false);
        printf("tokenize part with neither text nor bitmap rc = %d (expect 1)\n", rc);
        assert(rc == 1);

        parts[0] = &part_null_text;
        rc = mtmd_tokenize_from_parts(NULL, out, parts, 1, false);
        printf("tokenize part with null text pointer rc = %d (expect 1)\n", rc);
        assert(rc == 1);

        mtmd_bitmap_free(bmp);
        mtmd_input_chunks_free(out);
    }

    // Free the chunks
    mtmd_input_chunks_free(chunks);

    printf("\n\nDONE: test libmtmd C API...\n");

    return 0;
}
