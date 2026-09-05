#include "../src/llama-model-loader.h"
#include "../src/llama-model.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
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

// Writes a single, non-split GGUF file holding every tensor in `tensor_names`.
void write_single_file(const std::filesystem::path & path, const std::vector<const char *> & tensor_names) {
    gguf_context_ptr context(gguf_init_empty());
    require(context != nullptr, "could not allocate GGUF context");
    gguf_set_val_str(context.get(), "general.architecture", "llama");

    std::vector<ggml_tensor> tensors(tensor_names.size());
    for (size_t i = 0; i < tensor_names.size(); i++) {
        ggml_tensor & tensor = tensors[i];
        tensor      = ggml_tensor{};
        tensor.type = GGML_TYPE_F32;
        tensor.ne[0] = 1;
        tensor.ne[1] = 1;
        tensor.ne[2] = 1;
        tensor.ne[3] = 1;
        ggml_set_name(&tensor, tensor_names[i]);
        gguf_add_tensor(context.get(), &tensor);
    }
    require(gguf_write_to_file(context.get(), path.c_str(), true), "could not write GGUF metadata");

    // the freshly-built context never had its data-section offset resolved (that only
    // happens when reading a real file), so re-read the file we just wrote to learn the
    // actual on-disk offsets before writing each tensor's data at its own offset
    struct gguf_init_params read_params { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr written(gguf_init_from_file(path.c_str(), read_params));
    require(written != nullptr, "could not re-read GGUF metadata");

    FILE * file = std::fopen(path.c_str(), "r+b");
    require(file != nullptr, "could not open GGUF file for tensor data");
    for (size_t i = 0; i < tensor_names.size(); i++) {
        const int64_t tid = gguf_find_tensor(written.get(), tensor_names[i]);
        require(tid != -1, "tensor not found after writing GGUF metadata");
        const uint64_t offset = gguf_get_data_offset(written.get()) + gguf_get_tensor_offset(written.get(), tid);
        require(std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0, "could not seek to tensor data offset");
        const float value = static_cast<float>(i + 1);
        require(std::fwrite(&value, sizeof(value), 1, file) == 1, "could not write tensor data");
    }
    require(std::fclose(file) == 0, "could not close GGUF file");
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
    const std::string normalized_fname = std::filesystem::absolute(splits.front()).lexically_normal().string();
    for (std::string & split : splits) {
        split = std::filesystem::absolute(split).lexically_normal().string();
    }
    llama_model_loader loader(nullptr, nullptr, nullptr, normalized_fname, splits, nullptr, LLAMA_LOAD_MODE_MMAP, true,
                              true, false, nullptr, nullptr);
    std::filesystem::current_path(original_cwd);
    require(splits[0] == first.string(), "first split path was not normalized");
    require(splits[1] == second.string(), "second split path was not normalized");
    const auto * first_weight  = loader.get_weight("token_embd.weight");
    const auto * second_weight = loader.get_weight("output.weight");
    require(first_weight != nullptr && first_weight->idx == 0 && first_weight->offs > 0,
            "first tensor was not assigned shard zero");
    require(second_weight != nullptr && second_weight->idx == 1 && second_weight->offs > 0,
            "second tensor was not assigned shard one");

    // weights_map iterates in weight_name_comparer order (layer number, then name); neither
    // tensor has a "blk.N." prefix, so this falls back to plain name order: "output.weight"
    // sorts before "token_embd.weight" even though it is the *second* shard on disk.
    require(second_weight->ordinal == 0, "output.weight (shard 1) did not get the first ordinal");
    require(first_weight->ordinal == 1, "token_embd.weight (shard 0) did not get the second ordinal");

    llama_model_params params = llama_model_default_params();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_create(LLM_ARCH_LLAMA, params),
                                                                    &llama_model_free);
    require(model != nullptr, "could not create model source index");
    model->retain_tensor_sources(loader, std::move(splits));
    require(model->source_paths.size() == 2, "model did not retain both shard paths");
    require(model->source_paths[0] == first.string(), "model retained the wrong first source path");
    require(model->source_paths[1] == second.string(), "model retained the wrong second source path");
    require(model->tensor_sources.size() == 2, "model did not retain both tensor records");
    require(model->tensor_sources[0].name == "output.weight" && model->tensor_sources[1].name == "token_embd.weight",
            "model tensor records were not indexed by ordinal");

    // every captured tensor must resolve, by its own stamped ordinal (ordinal + 1, per
    // GGML_TENSOR_SRC_ORDINAL_NONE), to the source shard and offset that the loader
    // recorded for that same tensor
    const auto * first_source  = model->tensor_source_at(static_cast<int32_t>(first_weight->ordinal + 1));
    const auto * second_source = model->tensor_source_at(static_cast<int32_t>(second_weight->ordinal + 1));
    require(first_source != nullptr && first_source->name == "token_embd.weight" && first_source->source_idx == 0 &&
                first_source->offset == first_weight->offs && first_source->size == 4,
            "ordinal lookup returned the wrong source for the first-shard tensor");
    require(second_source != nullptr && second_source->name == "output.weight" && second_source->source_idx == 1 &&
                second_source->offset == second_weight->offs && second_source->size == 4,
            "ordinal lookup returned the wrong source for the split-shard tensor");

    require(model->tensor_source_at(GGML_TENSOR_SRC_ORDINAL_NONE) == nullptr,
            "ordinal lookup accepted the sentinel ordinal");
    require(model->tensor_source_at(-5) == nullptr, "ordinal lookup accepted a negative ordinal");
    require(model->tensor_source_at(static_cast<int32_t>(model->tensor_sources.size()) + 1) == nullptr,
            "ordinal lookup accepted an out-of-range stamped ordinal");

    require(model->source_path(0) != nullptr && *model->source_path(0) == first.string(),
            "source path lookup failed for shard zero");
    require(model->source_path(2) == nullptr, "out-of-bounds source path lookup did not fail");
    const uint16_t original_idx = loader.weights_map.at("output.weight").idx;
    loader.weights_map.at("output.weight").idx = 2;
    bool rejected_invalid_idx = false;
    try {
        model->retain_tensor_sources(loader, {first.string(), second.string()});
    } catch (const std::exception &) {
        rejected_invalid_idx = true;
    }
    require(rejected_invalid_idx, "out-of-bounds tensor source index was not rejected");
    loader.weights_map.at("output.weight").idx = original_idx;
    model->retain_tensor_sources(loader, {first.string(), second.string()});
    model->retain_tensor_sources(loader, {});
    require(model->source_paths.empty() && model->tensor_sources.empty(),
            "source-less load retained disk tensor sources");

    // retain_tensor_sources() cross-checks every tensors_by_name entry's stamped ordinal
    // against the source table it just built from the loader; that is the entire safety
    // argument for dropping name-based lookup, and llama_model_create() alone (without
    // load_tensors) leaves tensors_by_name empty, so it otherwise never runs. Populate it
    // directly, one case at a time, to exercise all three outcomes.
    {
        struct ggml_init_params ictx_params { /*.mem_size   =*/ ggml_tensor_overhead() * 4,
                                               /*.mem_buffer =*/ nullptr,
                                               /*.no_alloc   =*/ true };
        ggml_context_ptr ctx(ggml_init(ictx_params));
        require(ctx != nullptr, "could not allocate scratch ggml context");

        // correctly stamped: token_embd.weight carries its own ordinal
        ggml_tensor * correct = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_name(correct, "token_embd.weight");
        correct->src_ordinal = static_cast<int32_t>(first_weight->ordinal + 1);
        model->tensors_by_name = { { "token_embd.weight", correct } };
        bool threw_on_correct_stamp = false;
        try {
            model->retain_tensor_sources(loader, {first.string(), second.string()});
        } catch (const std::exception &) {
            threw_on_correct_stamp = true;
        }
        require(!threw_on_correct_stamp, "retain_tensor_sources rejected a correctly stamped tensor");
        model->tensors_by_name.clear();

        // wrong stamp: token_embd.weight carries output.weight's (otherwise valid) ordinal
        ggml_tensor * mislabeled = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_name(mislabeled, "token_embd.weight");
        mislabeled->src_ordinal = static_cast<int32_t>(second_weight->ordinal + 1);
        model->tensors_by_name = { { "token_embd.weight", mislabeled } };
        bool rejected_wrong_stamp = false;
        try {
            model->retain_tensor_sources(loader, {first.string(), second.string()});
        } catch (const std::exception &) {
            rejected_wrong_stamp = true;
        }
        require(rejected_wrong_stamp, "retain_tensor_sources accepted a tensor stamped with another tensor's ordinal");
        model->tensors_by_name.clear();

        // missed stamp: left at the sentinel even though the loader has a matching weight
        ggml_tensor * unstamped = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_name(unstamped, "token_embd.weight");
        unstamped->src_ordinal = GGML_TENSOR_SRC_ORDINAL_NONE;
        model->tensors_by_name = { { "token_embd.weight", unstamped } };
        bool rejected_missed_stamp = false;
        try {
            model->retain_tensor_sources(loader, {first.string(), second.string()});
        } catch (const std::exception &) {
            rejected_missed_stamp = true;
        }
        require(rejected_missed_stamp, "retain_tensor_sources accepted an unstamped tensor with a matching loader weight");
        model->tensors_by_name.clear();
    }

    // single, non-split shard: both tensors must still get distinct, dense ordinals and
    // resolve to source index 0 with the loader's own offsets
    const std::filesystem::path single = directory / "single.gguf";
    write_single_file(single, { "zeta.weight", "alpha.weight" });
    std::vector<std::string> no_splits;
    llama_model_loader single_loader(nullptr, nullptr, nullptr, single.string(), no_splits, nullptr,
                                     LLAMA_LOAD_MODE_MMAP, true, true, false, nullptr, nullptr);
    const auto * zeta_weight  = single_loader.get_weight("zeta.weight");
    const auto * alpha_weight = single_loader.get_weight("alpha.weight");
    require(zeta_weight != nullptr && zeta_weight->idx == 0 && zeta_weight->offs > 0,
            "zeta.weight was not assigned the single source shard");
    require(alpha_weight != nullptr && alpha_weight->idx == 0, "alpha.weight was not assigned the single source shard");
    require(alpha_weight->ordinal == 0 && zeta_weight->ordinal == 1,
            "single-file ordinals were not assigned in weights_map order");

    std::unique_ptr<llama_model, decltype(&llama_model_free)> single_model(llama_model_create(LLM_ARCH_LLAMA, params),
                                                                           &llama_model_free);
    require(single_model != nullptr, "could not create single-shard model source index");
    single_model->retain_tensor_sources(single_loader, { single.string() });
    require(single_model->tensor_sources.size() == 2, "single-shard model did not retain both tensor records");
    const auto * zeta_source  = single_model->tensor_source_at(static_cast<int32_t>(zeta_weight->ordinal + 1));
    const auto * alpha_source = single_model->tensor_source_at(static_cast<int32_t>(alpha_weight->ordinal + 1));
    require(zeta_source != nullptr && zeta_source->name == "zeta.weight" && zeta_source->source_idx == 0 &&
                zeta_source->offset == zeta_weight->offs && zeta_source->size == 4,
            "ordinal lookup returned the wrong source for the single-shard tensor 'zeta.weight'");
    require(alpha_source != nullptr && alpha_source->name == "alpha.weight" && alpha_source->source_idx == 0 &&
                alpha_source->offset == alpha_weight->offs && alpha_source->size == 4,
            "ordinal lookup returned the wrong source for the single-shard tensor 'alpha.weight'");

    std::filesystem::remove_all(directory, error);
    return 0;
}
