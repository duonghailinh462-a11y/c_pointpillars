#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <vector>
#include <string>

struct InferenceOutput {
    std::vector<float> bboxes;      // [batch, num_det, 7]
    std::vector<float> scores;      // [batch, num_det, num_classes]
    std::vector<int64_t> bbox_shape;
    std::vector<int64_t> score_shape;
};

class PythonInference {
public:
    PythonInference(const std::string& model_path);
    
    InferenceOutput run(
        const std::vector<float>& voxels,
        const std::vector<int>& coordinates,
        const std::vector<int>& num_points,
        int num_voxels
    );
    
private:
    std::string model_path_;
    
    std::string get_script_dir() const;
    std::string write_temp_file(const std::string& prefix, const std::vector<float>& data);
    std::string write_temp_file(const std::string& prefix, const std::vector<int>& data);
};

#endif // ONNX_INFERENCE_H
