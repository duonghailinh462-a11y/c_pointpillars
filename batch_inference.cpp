#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "voxelizer.h"
#include "onnx_inference.h"
#include "postprocess.h"

namespace fs = std::filesystem;

// Timing utilities
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

std::vector<float> load_kitti_data(const std::string& bin_file) {
    std::ifstream file(bin_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + bin_file);
    }
    
    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<float> points(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(points.data()), file_size);
    file.close();
    
    return points;
}

struct InferenceStats {
    double preprocess_time;
    double inference_time;
    double postprocess_time;
    double total_time;
    int num_detections;
};

InferenceStats process_frame(
    const std::string& bin_file,
    const std::string& onnx_model,
    float score_thr,
    float nms_thr,
    int max_num) {
    
    InferenceStats stats = {0, 0, 0, 0, 0};
    Timer total_timer;
    
    try {
        // Load and preprocess
        Timer preprocess_timer;
        auto points = load_kitti_data(bin_file);
        
        VoxelConfig voxel_config;
        Voxelizer voxelizer(voxel_config);
        auto voxel_data = voxelizer.generate(points);
        
        stats.preprocess_time = preprocess_timer.elapsed();
        
        // Inference
        Timer inference_timer;
        PythonInference inference(onnx_model);
        auto inference_output = inference.run(
            voxel_data.voxels,
            voxel_data.coordinates,
            voxel_data.num_points,
            voxel_data.num_voxels
        );
        stats.inference_time = inference_timer.elapsed();
        
        // Post-process
        Timer postprocess_timer;
        PostProcessor post_processor(score_thr, nms_thr, max_num);
        auto result = post_processor.process(
            inference_output.bboxes,
            inference_output.scores,
            inference_output.bbox_shape,
            inference_output.score_shape
        );
        stats.postprocess_time = postprocess_timer.elapsed();
        
        stats.num_detections = result.boxes_3d.size();
        stats.total_time = total_timer.elapsed();
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing " << bin_file << ": " << e.what() << std::endl;
        stats.total_time = total_timer.elapsed();
    }
    
    return stats;
}

int main(int argc, char* argv[]) {
    std::string data_dir = "/home/test/gw560_disk/zhw/PointDistiller/data/kitti/testing/velodyne";
    std::string onnx_model = "model/end2end_sim.onnx";
    float score_thr = 0.3f;
    float nms_thr = 0.01f;
    int max_num = 100;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--onnx-model" && i + 1 < argc) {
            onnx_model = argv[++i];
        } else if (arg == "--score-thr" && i + 1 < argc) {
            score_thr = std::stof(argv[++i]);
        } else if (arg == "--nms-thr" && i + 1 < argc) {
            nms_thr = std::stof(argv[++i]);
        } else if (arg == "--max-num" && i + 1 < argc) {
            max_num = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --data-dir <path>      Data directory (default: /home/test/gw560_disk/zhw/PointDistiller/data/kitti/testing/velodyne)" << std::endl;
            std::cout << "  --onnx-model <path>    ONNX model file (default: model/end2end_sim.onnx)" << std::endl;
            std::cout << "  --score-thr <float>    Score threshold (default: 0.3)" << std::endl;
            std::cout << "  --nms-thr <float>      NMS threshold (default: 0.01)" << std::endl;
            std::cout << "  --max-num <int>        Max detections (default: 100)" << std::endl;
            return 0;
        }
    }
    
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Batch Inference - KITTI Point Cloud" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Data directory: " << data_dir << std::endl;
    std::cout << "  ONNX model: " << onnx_model << std::endl;
    std::cout << "  Score threshold: " << score_thr << std::endl;
    std::cout << "  NMS threshold: " << nms_thr << std::endl;
    std::cout << "  Max detections: " << max_num << std::endl;
    
    // Check if directory exists
    if (!fs::exists(data_dir)) {
        std::cerr << "Error: Data directory not found: " << data_dir << std::endl;
        return 1;
    }
    
    if (!fs::exists(onnx_model)) {
        std::cerr << "Error: ONNX model not found: " << onnx_model << std::endl;
        return 1;
    }
    
    // Collect all .bin files
    std::vector<std::string> bin_files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            bin_files.push_back(entry.path().string());
        }
    }
    
    std::sort(bin_files.begin(), bin_files.end());
    
    std::cout << "\nFound " << bin_files.size() << " .bin files" << std::endl;
    
    if (bin_files.empty()) {
        std::cerr << "No .bin files found in " << data_dir << std::endl;
        return 1;
    }
    
    // Process all frames
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Processing frames..." << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::vector<InferenceStats> all_stats;
    Timer total_timer;
    
    for (size_t i = 0; i < bin_files.size(); ++i) {
        const auto& bin_file = bin_files[i];
        std::string filename = fs::path(bin_file).filename().string();
        
        std::cout << "[" << std::setw(3) << (i + 1) << "/" << bin_files.size() << "] "
                  << filename << " ... ";
        std::cout.flush();
        
        auto stats = process_frame(bin_file, onnx_model, score_thr, nms_thr, max_num);
        all_stats.push_back(stats);
        
        std::cout << std::fixed << std::setprecision(2)
                  << stats.total_time << " ms ("
                  << stats.num_detections << " detections)" << std::endl;
    }
    
    double total_elapsed = total_timer.elapsed();
    
    // Calculate statistics
    std::vector<double> preprocess_times, inference_times, postprocess_times, total_times;
    int total_detections = 0;
    
    for (const auto& stats : all_stats) {
        preprocess_times.push_back(stats.preprocess_time);
        inference_times.push_back(stats.inference_time);
        postprocess_times.push_back(stats.postprocess_time);
        total_times.push_back(stats.total_time);
        total_detections += stats.num_detections;
    }
    
    auto calc_stats = [](const std::vector<double>& times) {
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double avg = sum / times.size();
        double min = *std::min_element(times.begin(), times.end());
        double max = *std::max_element(times.begin(), times.end());
        return std::make_tuple(avg, min, max, sum);
    };
    
    auto [preprocess_avg, preprocess_min, preprocess_max, preprocess_sum] = calc_stats(preprocess_times);
    auto [inference_avg, inference_min, inference_max, inference_sum] = calc_stats(inference_times);
    auto [postprocess_avg, postprocess_min, postprocess_max, postprocess_sum] = calc_stats(postprocess_times);
    auto [total_avg, total_min, total_max, total_sum] = calc_stats(total_times);
    
    // Print results
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Inference Statistics" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\nPreprocessing (Voxelization):" << std::endl;
    std::cout << "  Average: " << std::setw(8) << preprocess_avg << " ms" << std::endl;
    std::cout << "  Min:     " << std::setw(8) << preprocess_min << " ms" << std::endl;
    std::cout << "  Max:     " << std::setw(8) << preprocess_max << " ms" << std::endl;
    std::cout << "  Total:   " << std::setw(8) << preprocess_sum << " ms" << std::endl;
    
    std::cout << "\nInference (ONNX Model):" << std::endl;
    std::cout << "  Average: " << std::setw(8) << inference_avg << " ms" << std::endl;
    std::cout << "  Min:     " << std::setw(8) << inference_min << " ms" << std::endl;
    std::cout << "  Max:     " << std::setw(8) << inference_max << " ms" << std::endl;
    std::cout << "  Total:   " << std::setw(8) << inference_sum << " ms" << std::endl;
    
    std::cout << "\nPostprocessing (NMS Filtering):" << std::endl;
    std::cout << "  Average: " << std::setw(8) << postprocess_avg << " ms" << std::endl;
    std::cout << "  Min:     " << std::setw(8) << postprocess_min << " ms" << std::endl;
    std::cout << "  Max:     " << std::setw(8) << postprocess_max << " ms" << std::endl;
    std::cout << "  Total:   " << std::setw(8) << postprocess_sum << " ms" << std::endl;
    
    std::cout << "\nTotal Per Frame:" << std::endl;
    std::cout << "  Average: " << std::setw(8) << total_avg << " ms" << std::endl;
    std::cout << "  Min:     " << std::setw(8) << total_min << " ms" << std::endl;
    std::cout << "  Max:     " << std::setw(8) << total_max << " ms" << std::endl;
    std::cout << "  Total:   " << std::setw(8) << total_sum << " ms" << std::endl;
    
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Frames processed: " << bin_files.size() << std::endl;
    std::cout << "  Total detections: " << total_detections << std::endl;
    std::cout << "  Avg detections/frame: " << std::fixed << std::setprecision(1)
              << (double)total_detections / bin_files.size() << std::endl;
    std::cout << "  Wall clock time: " << std::fixed << std::setprecision(2)
              << total_elapsed << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
              << (bin_files.size() * 1000.0 / total_elapsed) << " frames/sec" << std::endl;
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    
    return 0;
}
