#include "../src/llama-model-loader.h"
#include "../src/llama-model.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-model-tensor-sources: %s\n", message);
        std::abort();
    }
}

void write_shard(const std::filesystem::path & path, uint16_t split_no, const char * tensor_name) {
    gguf_context_ptr context(gguf_init_empty());
    require(context != nullptr, "could not allocate GGUF context");
    gguf_set_val_str(context.get(), "general.architecture", "llama");
    gguf_set_val_u16(context.get(), "split.count", 2);
    gguf_set_val_u16(context.get(), "split.no", split_no);
    gguf_set_val_i32(context.get(), "split.tensors.count", 2);

    ggml_tensor tensor{};
    tensor.type  = GGML_TYPE_F32;
    tensor.ne[0] = 1;
    tensor.ne[1] = 1;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    ggml_set_name(&tensor, tensor_name);
    gguf_add_tensor(context.get(), &tensor);
    require(gguf_write_to_file(context.get(), path.c_str(), true), "could not write GGUF metadata");

    FILE * file = std::fopen(path.c_str(), "ab");
    require(file != nullptr, "could not append GGUF tensor data");
    const float value = static_cast<float>(split_no + 1);
    require(std::fwrite(&value, sizeof(value), 1, file) == 1, "could not write tensor data");
    require(std::fclose(file) == 0, "could not close GGUF shard");
}

}  // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::weakly_canonical(std::filesystem::temp_directory_path()) /
        ("tc-model-tensor-sources-" + std::to_string(static_cast<unsigned long long>(::getpid())));
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    require(std::filesystem::create_directory(directory, error), "could not create temporary directory");

    const std::filesystem::path first  = directory / "tiny-00001-of-00002.gguf";
    const std::filesystem::path second = directory / "tiny-00002-of-00002.gguf";
    write_shard(first, 0, "token_embd.weight");
    write_shard(second, 1, "output.weight");

    const std::filesystem::path original_cwd = std::filesystem::current_path();
    std::filesystem::current_path(directory);
    std::vector<std::string> splits{ first.filename().string(), second.filename().string() };
    llama_model_loader loader(nullptr, nullptr, nullptr, first.filename().string(), splits, nullptr, LLAMA_LOAD_MODE_MMAP, true,
                              true, false, nullptr, nullptr);
    std::filesystem::current_path(original_cwd);
    require(loader.source_paths.size() == 2, "loader did not retain both shard paths");
    require(loader.source_paths[0] == first.string(), "first source path was not retained");
    require(loader.source_paths[1] == second.string(), "second source path was not retained");
    const auto * first_weight  = loader.get_weight("token_embd.weight");
    const auto * second_weight = loader.get_weight("output.weight");
    require(first_weight != nullptr && first_weight->idx == 0 && first_weight->offs > 0,
            "first tensor was not assigned shard zero");
    require(second_weight != nullptr && second_weight->idx == 1 && second_weight->offs > 0,
            "second tensor was not assigned shard one");

    llama_model_params params = llama_model_default_params();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_create(LLM_ARCH_LLAMA, params),
                                                                    &llama_model_free);
    require(model != nullptr, "could not create model source index");
    model->retain_tensor_sources(loader);
    require(loader.source_paths.empty(), "model did not take ownership of source paths");
    require(model->source_paths.size() == 2, "model did not retain both shard paths");
    require(model->source_paths[0] == first.string(), "model retained the wrong first source path");
    require(model->source_paths[1] == second.string(), "model retained the wrong second source path");
    require(model->tensor_sources.size() == 2, "model did not retain both tensor records");
    require(model->tensor_sources[0].name == "output.weight" && model->tensor_sources[1].name == "token_embd.weight",
            "model tensor records were not sorted by name");
    const auto * first_source = model->tensor_source("token_embd.weight");
    const auto * second_source = model->tensor_source("output.weight");
    require(first_source != nullptr && first_source->source_idx == 0 && first_source->offset == first_weight->offs &&
                first_source->size == 4,
            "binary lookup returned the wrong first tensor source");
    require(second_source != nullptr && second_source->source_idx == 1 &&
                second_source->offset == second_weight->offs && second_source->size == 4,
            "binary lookup returned the wrong second tensor source");
    require(model->tensor_source("missing.weight") == nullptr, "binary lookup returned a missing tensor source");
    require(model->tensor_source(nullptr) == nullptr, "null tensor lookup did not return null");
    require(model->source_path(0) != nullptr && *model->source_path(0) == first.string(),
            "source path lookup failed for shard zero");
    require(model->source_path(2) == nullptr, "out-of-bounds source path lookup did not fail");

    std::filesystem::remove_all(directory, error);
    return 0;
}
