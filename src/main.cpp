#include "llama.h"

#include <iostream>
#include <vector>
#include <string>

int main()
{
    std::cout << "Starting Gemma...\n";

    // --------------------------------------------------
    // 1. Initialize llama.cpp
    // --------------------------------------------------

    llama_backend_init();

    // --------------------------------------------------
    // 2. Select the RTX 3050
    // --------------------------------------------------

    ggml_backend_dev_t nvidia_gpu = ggml_backend_dev_get(1);

    ggml_backend_dev_t devices[] = {
        nvidia_gpu,
        nullptr};

    std::cout << "Using GPU: "
              << ggml_backend_dev_description(nvidia_gpu)
              << "\n";

    // --------------------------------------------------
    // 3. Configure the model
    // --------------------------------------------------

    llama_model_params model_params = llama_model_default_params();

    model_params.n_gpu_layers = -1;
    model_params.devices = devices;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.main_gpu = 0;

    // --------------------------------------------------
    // 4. Load Gemma
    // --------------------------------------------------

    const char *model_path =
        "D:/padhai likhai/Programming/gemma_project/models/gemma-4-E4B-it-Q4_K_M.gguf";

    llama_model *model =
        llama_model_load_from_file(model_path, model_params);

    if (model == nullptr)
    {
        std::cerr << "Failed to load Gemma!\n";
        llama_backend_free();
        return 1;
    }

    std::cout << "Gemma loaded successfully!\n";

    // --------------------------------------------------
    // 5. Create an inference context
    // --------------------------------------------------

    llama_context_params ctx_params =
        llama_context_default_params();

    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;

    llama_context *ctx =
        llama_init_from_model(model, ctx_params);

    if (ctx == nullptr)
    {
        std::cerr << "Failed to create context!\n";

        llama_model_free(model);
        llama_backend_free();

        return 1;
    }

    std::cout << "Context created successfully!\n";

    // --------------------------------------------------
    // 6. Get the model vocabulary
    // --------------------------------------------------

    const llama_vocab *vocab =
        llama_model_get_vocab(model);

    // --------------------------------------------------
    // 7. Tokenize our prompt
    // --------------------------------------------------

    std::string prompt = "Hello! Introduce yourself in one sentence.";

    std::vector<llama_token> tokens(prompt.size() + 32);

    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        static_cast<int32_t>(prompt.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false);

    if (n_tokens < 0)
    {
        std::cerr << "Tokenization failed.\n";

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();

        return 1;
    }

    tokens.resize(n_tokens);

    std::cout << "Prompt tokens: "
              << tokens.size()
              << "\n";

    // --------------------------------------------------
    // 8. Create a batch containing our tokens
    // --------------------------------------------------

    llama_batch batch = llama_batch_init(
        static_cast<int32_t>(tokens.size()),
        0,
        1);

    for (size_t i = 0; i < tokens.size(); i++)
    {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;

        // We only need logits for the final prompt token.
        batch.logits[i] = (i == tokens.size() - 1);
    }

    batch.n_tokens = static_cast<int32_t>(tokens.size());

    // --------------------------------------------------
    // 9. Run the prompt through Gemma
    // --------------------------------------------------

    if (llama_decode(ctx, batch) != 0)
    {
        std::cerr << "llama_decode() failed.\n";

        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();

        return 1;
    }

    std::cout << "Prompt processed successfully!\n";

    // --------------------------------------------------
    // 10. Create a sampler
    // --------------------------------------------------

    llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();

    llama_sampler *sampler =
        llama_sampler_chain_init(sampler_params);

    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_temp(0.8f));

    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_dist(1234));

    // --------------------------------------------------
    // 11. Generate tokens
    // --------------------------------------------------

    std::cout << "\nGemma: ";

    for (int i = 0; i < 100; i++)
    {
        llama_token new_token =
            llama_sampler_sample(sampler, ctx, -1);

        // Stop if Gemma produces an end-of-generation token.
        if (llama_vocab_is_eog(vocab, new_token))
        {
            break;
        }

        char buffer[256];

        int32_t length = llama_token_to_piece(
            vocab,
            new_token,
            buffer,
            sizeof(buffer),
            0,
            true);

        if (length > 0)
        {
            std::cout.write(buffer, length);
            std::cout.flush();
        }

        // Prepare the next token.
        batch.n_tokens = 1;

        batch.token[0] = new_token;
        batch.pos[0] = static_cast<llama_pos>(tokens.size() + i);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx, batch) != 0)
        {
            std::cerr << "\nGeneration failed.\n";
            break;
        }
    }

    std::cout << "\n";

    // --------------------------------------------------
    // 12. Cleanup
    // --------------------------------------------------

    llama_sampler_free(sampler);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}