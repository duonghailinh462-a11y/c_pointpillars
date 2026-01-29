#!/usr/bin/env python
"""
ONNX Model Inference Service for C++ Integration

This module provides a simple interface for C++ to call Python inference.
It handles ONNX model loading and inference with voxelized input.
"""

import argparse
import json
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path


class InferenceService:
    """Inference service for PointPillars ONNX model."""
    
    def __init__(self, onnx_model):
        """Initialize ONNX Runtime session."""
        print(f"Loading ONNX model: {onnx_model}", file=sys.stderr)
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model, sess_options, providers=providers)
        
        print("✓ ONNX session created successfully!", file=sys.stderr)
    
    def run_inference(self, voxels_data, coors_data, num_points_data):
        """
        Run inference on voxelized data.
        
        Args:
            voxels_data: Flattened voxels array [num_voxels * 32 * 4]
            coors_data: Flattened coordinates array [num_voxels * 4]
            num_points_data: Flattened num_points array [num_voxels]
        
        Returns:
            Dictionary with 'bboxes' and 'scores' keys
        """
        print(f"Running ONNX inference...", file=sys.stderr)
        
        # Reshape input data
        num_voxels = len(num_points_data)
        
        voxels = voxels_data.reshape(num_voxels, 32, 4).astype(np.float32)
        coors = coors_data.reshape(num_voxels, 4).astype(np.int32)
        num_points = num_points_data.reshape(num_voxels).astype(np.int32)
        
        print(f"  Voxels shape: {voxels.shape}", file=sys.stderr)
        print(f"  Coordinates shape: {coors.shape}", file=sys.stderr)
        print(f"  Num points shape: {num_points.shape}", file=sys.stderr)
        
        # Prepare input dict
        input_dict = {
            'voxels': voxels,
            'num_points': num_points,
            'coors': coors
        }
        
        # Run inference
        outputs = self.session.run(None, input_dict)
        
        print(f"✓ Inference completed!", file=sys.stderr)
        print(f"  Bboxes shape: {outputs[0].shape}", file=sys.stderr)
        print(f"  Scores shape: {outputs[1].shape}", file=sys.stderr)
        
        return {
            'bboxes': outputs[0].tolist(),
            'scores': outputs[1].tolist(),
            'bbox_shape': list(outputs[0].shape),
            'score_shape': list(outputs[1].shape)
        }


def main():
    """Command-line interface for inference service."""
    parser = argparse.ArgumentParser(description='ONNX Inference Service')
    parser.add_argument('--onnx-model', required=True, help='ONNX model file path')
    parser.add_argument('--voxels', required=True, help='Voxels data file (binary)')
    parser.add_argument('--coors', required=True, help='Coordinates data file (binary)')
    parser.add_argument('--num-points', required=True, help='Num points data file (binary)')
    
    args = parser.parse_args()
    
    # Load data
    voxels_data = np.fromfile(args.voxels, dtype=np.float32)
    coors_data = np.fromfile(args.coors, dtype=np.int32)
    num_points_data = np.fromfile(args.num_points, dtype=np.int32)
    
    # Run inference
    service = InferenceService(args.onnx_model)
    result = service.run_inference(voxels_data, coors_data, num_points_data)
    
    # Output as JSON
    print(json.dumps(result))


if __name__ == '__main__':
    main()
