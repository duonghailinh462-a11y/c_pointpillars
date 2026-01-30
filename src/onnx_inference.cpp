#include "onnx_inference.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <sstream>
#include <regex>

PythonInference::PythonInference(const std::string& model_path) 
    : model_path_(model_path) {
    std::cout << "Initializing Python inference service..." << std::endl;
    std::cout << "✓ Python inference service ready!" << std::endl;
}

std::string PythonInference::get_script_dir() const {
    // Get the directory of this executable
    // For now, assume inference_service.py is in the same directory
    return ".";
}

std::string PythonInference::write_temp_file(const std::string& prefix, const std::vector<float>& data) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    
    std::string filename = "/tmp/" + prefix + "_" + std::to_string(micros) + ".bin";
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create temp file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    file.close();
    return filename;
}

std::string PythonInference::write_temp_file(const std::string& prefix, const std::vector<int>& data) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    
    std::string filename = "/tmp/" + prefix + "_" + std::to_string(micros) + ".bin";
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create temp file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int));
    file.close();
    return filename;
}

InferenceOutput PythonInference::run(
    const std::vector<float>& voxels,
    const std::vector<int>& coordinates,
    const std::vector<int>& num_points,
    int num_voxels) {
    
    std::cout << "Running Python inference..." << std::endl;
    
    InferenceOutput output;
    
    try {
        // Write temporary files
        std::string voxels_file = write_temp_file("voxels", voxels);
        std::string coors_file = write_temp_file("coors", coordinates);
        std::string num_pts_file = write_temp_file("num_points", num_points);
        
        // Build Python command
        std::string script_dir = get_script_dir();
        std::string cmd = "python " + script_dir + "/inference_service.py"
            " --onnx-model " + model_path_ +
            " --voxels " + voxels_file +
            " --coors " + coors_file +
            " --num-points " + num_pts_file;
        
        std::cout << "  Executing: " << cmd << std::endl;
        
        // Execute Python script and capture output
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to execute Python inference");
        }
        
        // Read JSON output
        std::string json_output;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            json_output += buffer;
        }
        pclose(pipe);
        
        // Simple JSON parsing using regex
        // Extract bboxes array
        std::regex bboxes_regex(R"("bboxes"\s*:\s*\[([\d.,\s-]+)\])");
        std::regex scores_regex(R"("scores"\s*:\s*\[([\d.,\s-]+)\])");
        std::regex bbox_shape_regex(R"("bbox_shape"\s*:\s*\[([\d,\s]+)\])");
        std::regex score_shape_regex(R"("score_shape"\s*:\s*\[([\d,\s]+)\])");
        std::regex number_regex(R"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)");
        
        std::smatch match;
        
        // Extract bboxes
        if (std::regex_search(json_output, match, bboxes_regex)) {
            std::string numbers_str = match[1];
            std::sregex_iterator iter(numbers_str.begin(), numbers_str.end(), number_regex);
            std::sregex_iterator end;
            while (iter != end) {
                output.bboxes.push_back(std::stof(iter->str()));
                ++iter;
            }
        }
        
        // Extract scores
        if (std::regex_search(json_output, match, scores_regex)) {
            std::string numbers_str = match[1];
            std::sregex_iterator iter(numbers_str.begin(), numbers_str.end(), number_regex);
            std::sregex_iterator end;
            while (iter != end) {
                output.scores.push_back(std::stof(iter->str()));
                ++iter;
            }
        }
        
        // Extract bbox_shape
        if (std::regex_search(json_output, match, bbox_shape_regex)) {
            std::string numbers_str = match[1];
            std::sregex_iterator iter(numbers_str.begin(), numbers_str.end(), number_regex);
            std::sregex_iterator end;
            while (iter != end) {
                output.bbox_shape.push_back(std::stoll(iter->str()));
                ++iter;
            }
        }
        
        // Extract score_shape
        if (std::regex_search(json_output, match, score_shape_regex)) {
            std::string numbers_str = match[1];
            std::sregex_iterator iter(numbers_str.begin(), numbers_str.end(), number_regex);
            std::sregex_iterator end;
            while (iter != end) {
                output.score_shape.push_back(std::stoll(iter->str()));
                ++iter;
            }
        }
        
        std::cout << "✓ Inference completed!" << std::endl;
        std::cout << "  Bboxes shape: [";
        for (size_t i = 0; i < output.bbox_shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << output.bbox_shape[i];
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  Scores shape: [";
        for (size_t i = 0; i < output.score_shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << output.score_shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Clean up temp files
        std::remove(voxels_file.c_str());
        std::remove(coors_file.c_str());
        std::remove(num_pts_file.c_str());
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Inference error: " << e.what() << std::endl;
        throw;
    }
    
    return output;
}
