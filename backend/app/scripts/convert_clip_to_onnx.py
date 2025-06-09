import torch
import open_clip
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import argparse

def convert_model(model_name, pretrained, output_dir):
    """
    지정된 CLIP 모델을 ONNX 형식으로 변환하고 양자화합니다.
    """
    print(f"모델 변환 시작: {model_name} (pretrained: {pretrained})")
    print(f"출력 디렉토리: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 원본 CLIP 모델 로드
    device = "cpu"  # CPU 기반 변환
    print("1/5: 원본 CLIP 모델 로드 중...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        print("   ... 로드 완료")
    except Exception as e:
        print(f"   ... 모델 로드 실패: {e}")
        return

    # 2. ONNX로 변환
    # 2-1. Visual (이미지) 모델 변환
    visual_output_path = os.path.join(output_dir, "clip_visual.onnx")
    print(f"2/5: Visual 모델을 ONNX로 변환 중... -> {visual_output_path}")
    try:
        image_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model.visual,
            image_input,
            visual_output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("   ... Visual 모델 변환 완료")
    except Exception as e:
        print(f"   ... Visual 모델 변환 실패: {e}")
        return

    # 2-2. Text (텍스트) 모델 변환
    text_output_path = os.path.join(output_dir, "clip_text.onnx")
    print(f"3/5: Text 모델을 ONNX로 변환 중... -> {text_output_path}")
    try:
        text_input = open_clip.tokenize(["a diagram of a sentence"])
        torch.onnx.export(
            model,
            text_input,
            text_output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("   ... Text 모델 변환 완료")
    except Exception as e:
        print(f"   ... Text 모델 변환 실패: {e}")
        # Visual 모델 변환은 성공했을 수 있으므로 계속 진행
        pass

    # 3. 동적 양자화 (Dynamic Quantization) 적용
    # 3-1. Visual 모델 양자화
    visual_quant_output_path = os.path.join(output_dir, "clip_visual.quant.onnx")
    print(f"4/5: Visual ONNX 모델 양자화 중... -> {visual_quant_output_path}")
    try:
        # op_types_to_quantize를 명시하여 Conv 연산자 양자화를 제외합니다.
        # ConvInteger 연산자 미지원 런타임 환경과의 호환성을 위함입니다.
        quantize_dynamic(
            model_input=visual_output_path,
            model_output=visual_quant_output_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=['MatMul', 'Attention', 'Gather', 'Transpose', 'EmbedLayerNormalization']
        )
        os.remove(visual_output_path)  # 원본 ONNX 파일 삭제
        print("   ... Visual 모델 양자화 완료 (Conv 제외)")
    except Exception as e:
        print(f"   ... Visual 모델 양자화 실패: {e}")

    # 3-2. Text 모델 양자화
    text_quant_output_path = os.path.join(output_dir, "clip_text.quant.onnx")
    print(f"5/5: Text ONNX 모델 양자화 중... -> {text_quant_output_path}")
    if os.path.exists(text_output_path):
        try:
            quantize_dynamic(
                model_input=text_output_path,
                model_output=text_quant_output_path,
                weight_type=QuantType.QInt8
            )
            os.remove(text_output_path)  # 원본 ONNX 파일 삭제
            print("   ... Text 모델 양자화 완료")
        except Exception as e:
            print(f"   ... Text 모델 양자화 실패: {e}")
    else:
        print("   ... Text ONNX 파일이 없어 양자화를 건너뜁니다.")


    print("\n모델 변환 및 양자화 작업이 모두 완료되었습니다.")
    print(f"최종 모델 파일은 '{output_dir}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CLIP models to ONNX and quantize them.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="The name of the CLIP model to convert (e.g., 'ViT-B-32')."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="The pretrained dataset name for the CLIP model (e.g., 'laion2b_s34b_b79k')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/clip_onnx",
        help="The directory to save the converted ONNX models."
    )
    args = parser.parse_args()

    convert_model(args.model_name, args.pretrained, args.output_dir) 