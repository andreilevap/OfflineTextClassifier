$models = @(
    @{ Path = "Models\model.onnx"; Url = "https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli/resolve/main/onnx/model.onnx" },
    @{ Path = "Models\model_quantized.onnx"; Url = "https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli/blob/main/onnx/model_quantized.onnx" },
    @{ Path = "Models\spm.model"; Url = "https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli/resolve/main/spm.model" }
)

foreach ($model in $models) {
    if (-Not (Test-Path $model.Path)) {
        Write-Host "Downloading $($model.Path)..."
        Invoke-WebRequest -Uri $model.Url -OutFile $model.Path
    }
}