#include <opencv2/opencv.hpp>
#include <json.hpp>
#include <CLI11.hpp>

#include "common/types.h"
#include "common/logger.h"
#include "common/canvas.h"
#include "core/pipeline.h"
#include "output/json_writer.h"
#include "inject/inject.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    vinylizer::Logger::instance().set_level(vinylizer::LogLevel::INFO);

    CLI::App app{"Vinylizer C++ - FH6 Livery Vectorizer", "vinylizer"};

    // ── Subcommands ──────────────────────────────────────────
    auto* opt_cmd = app.add_subcommand("optimize", "Run vectorization optimization");
    auto* inject_cmd = app.add_subcommand("inject", "Inject shapes into FH6 memory");

    // ── Optimize options ─────────────────────────────────────
    std::string input_path;
    std::string output_path = "output.json";
    int num_shapes = 500;
    int canvas_w = 512;
    int canvas_h = 512;
    int num_cycles = 3;
    std::string lr_str = "0.1,0.01";

    opt_cmd->add_option("-i,--input", input_path, "Input image path")->required()->check(CLI::ExistingFile);
    opt_cmd->add_option("-o,--output", output_path, "Output JSON path");
    opt_cmd->add_option("-n,--num-shapes", num_shapes, "Number of shapes")->default_val("500");
    opt_cmd->add_option("--width", canvas_w, "Canvas width")->default_val("512");
    opt_cmd->add_option("--height", canvas_h, "Canvas height")->default_val("512");
    opt_cmd->add_option("-c,--cycles", num_cycles, "Number of optimization cycles")->default_val("3");
    opt_cmd->add_option("--lr", lr_str, "Learning rate schedule (start,end)")->default_val("0.1,0.01");

    // ── Inject options ───────────────────────────────────────
    std::string json_path;
    int pid = 0;
    int layer_count = 0;

    inject_cmd->add_option("-j,--json", json_path, "JSON file path")->required()->check(CLI::ExistingFile);
    inject_cmd->add_option("-l,--layers", layer_count, "Number of placeholder layers in FH6")->required();
    inject_cmd->add_option("-p,--pid", pid, "FH6 process ID (0=auto)")->default_val("0");

    app.require_subcommand(1);

    CLI11_PARSE(app, argc, argv);

    if (opt_cmd->parsed()) {
        vinylizer::PipelineConfig config;
        config.input_path = input_path;
        config.output_path = output_path;
        config.num_shapes = num_shapes;
        config.canvas_w = canvas_w;
        config.canvas_h = canvas_h;
        config.num_cycles = num_cycles;

        // Parse lr schedule
        auto comma = lr_str.find(',');
        if (comma != std::string::npos) {
            config.lr_start = std::stof(lr_str.substr(0, comma));
            config.lr_end = std::stof(lr_str.substr(comma + 1));
        }

        VIN_INFO("Starting optimization: input=%s, shapes=%d, cycles=%d",
                 input_path.c_str(), num_shapes, num_cycles);

        vinylizer::run_pipeline(config);

        VIN_INFO("Optimization complete: output=%s", output_path.c_str());

    } else if (inject_cmd->parsed()) {
        vinylizer::InjectConfig config;
        config.json_path = json_path;
        config.process_id = pid;
        config.layer_count = layer_count;

        VIN_INFO("Injecting shapes: json=%s, pid=%d", json_path.c_str(), pid);

        if (vinylizer::inject_shapes(config)) {
            VIN_INFO("Injection successful");
        } else {
            VIN_ERROR("Injection failed");
            return 1;
        }
    }

    return 0;
}
