import os
import sys
import unittest
from PIL import Image
import torch
from vlmdrive.vlm.base_vlm_planner import BaseVLMWaypointPlanner

class TestVLMInference(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_image_path = "tests/test_data/test_image.jpg"
        self.test_prompt = "Describe what you see in this image."
        
        # Ensure test image exists
        if not os.path.exists(self.test_image_path):
            os.makedirs(os.path.dirname(self.test_image_path), exist_ok=True)
            # Create a simple test image
            img = Image.new('RGB', (224, 224), color='red')
            img.save(self.test_image_path)

    def test_llava_inference(self):
        """Test LLaVA model inference"""
        try:
            planner = BaseVLMWaypointPlanner("llava")
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_default_tensor_type('torch.cuda.FloatTensor' if device == 'cuda' else 'torch.FloatTensor')
            
            # Move model to device
            planner.model = planner.model.to(device)
            
            result = planner.vlm_inference(
                text=self.test_prompt,
                images=self.test_image_path,
                processor=planner.processor,
                model=planner.model,
                tokenizer=planner.tokenizer
            )
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            print(f"LLaVA output: {result}")
        except Exception as e:
            self.fail(f"LLaVA test failed with error: {str(e)}")

    def test_qwen2_vl_inference(self):
        """Test Qwen2-VL model inference"""
        try:
            planner = BaseVLMWaypointPlanner("qwen2_vl")
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_default_tensor_type('torch.cuda.FloatTensor' if device == 'cuda' else 'torch.FloatTensor')
            
            # Move model to device
            planner.model = planner.model.to(device)
            
            result = planner.vlm_inference(
                text=self.test_prompt,
                images=self.test_image_path,
                processor=planner.processor,
                model=planner.model,
                tokenizer=planner.tokenizer
            )
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            print(f"Qwen2-VL output: {result}")
        except Exception as e:
            self.fail(f"Qwen2-VL test failed with error: {str(e)}")

    def test_qwen2_5_vl_inference(self):
        """Test Qwen2.5-VL model inference"""
        try:
            planner = BaseVLMWaypointPlanner("qwen2.5_vl")
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_default_tensor_type('torch.cuda.FloatTensor' if device == 'cuda' else 'torch.FloatTensor')
            
            # Move model to device
            planner.model = planner.model.to(device)
            
            result = planner.vlm_inference(
                text=self.test_prompt,
                images=self.test_image_path,
                processor=planner.processor,
                model=planner.model,
                tokenizer=planner.tokenizer
            )
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            print(f"Qwen2.5-VL output: {result}")
        except Exception as e:
            self.fail(f"Qwen2.5-VL test failed with error: {str(e)}")

    def test_gpt_inference(self):
        """Test GPT model inference with single image"""
        try:
            planner = BaseVLMWaypointPlanner("gpt")
            
            result = planner.vlm_inference(
                text=self.test_prompt,
                images=self.test_image_path,
                processor=None,  # GPT implementation doesn't use these
                model=None,
                tokenizer=None
            )
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            print(f"GPT single image output: {result}")
        except Exception as e:
            self.fail(f"GPT single image test failed with error: {str(e)}")
            
    def test_gpt_multi_images_inference(self):
        """Test GPT model inference with multiple images"""
        try:
            planner = BaseVLMWaypointPlanner("api")
            
            # 创建第二个测试图片
            test_image_path2 = "tests/test_data/test_image2.jpg"
            if not os.path.exists(test_image_path2):
                os.makedirs(os.path.dirname(test_image_path2), exist_ok=True)
                img = Image.new('RGB', (224, 224), color='blue')
                img.save(test_image_path2)
            
            result = planner.vlm_inference(
                text="Compare these two images.",
                images=[self.test_image_path, test_image_path2],
                processor=None,
                model=None,
                tokenizer=None
            )
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
            print(f"GPT multiple images output: {result}")
        except Exception as e:
            self.fail(f"GPT multiple images test failed with error: {str(e)}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行VLM推理测试')
    parser.add_argument('--model', type=str, choices=['llava', 'qwen2_vl', 'qwen2_5_vl', 'gpt'],
                        help='指定要测试的模型: llava, qwen2_vl, qwen2_5_vl 或 gpt')
    parser.add_argument('--multi-images', action='store_true',
                        help='是否运行多图片测试')
    args = parser.parse_args()
    
    if args.model:
        # 创建测试套件
        suite = unittest.TestSuite()
        # 根据参数选择测试用例
        if args.model == 'gpt' and args.multi_images:
            test_name = f'test_{args.model}_multi_images_inference'
        else:
            test_name = f'test_{args.model}_inference'
        suite.addTest(TestVLMInference(test_name))
        # 运行测试套件
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # 如果没有指定模型，运行所有测试
        unittest.main(verbosity=2)
