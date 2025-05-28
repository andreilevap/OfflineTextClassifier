
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System.Reflection;
using Sentencepiece;


namespace LevapTech.OfflineTextClassifier
{
    public record ClassificationResult(string Label, float Confidence);

    public class OfflineTextClasifier : IDisposable
    {
        private const string ModelLightResource = "model_quantized.onnx";
        private const string ModelResource = "model.onnx";
        private const string TokenizerResource = "spm.model";

        private readonly InferenceSession _session;
        private readonly SentencePieceTokenizer _tokenizer;
        public bool IsLightweight { get; }
        Stream _spmModelStream;

        public OfflineTextClasifier(bool isLightweight = false)
        {
            IsLightweight = isLightweight;

            var assembly = Assembly.GetExecutingAssembly();

            var modelResource = !isLightweight
                ? ModelResource
                : ModelLightResource;


            var modelPath = Path.Combine(AppContext.BaseDirectory, "Models", modelResource);
            _session = new InferenceSession(modelPath);

            var spmModelPath = Path.Combine(AppContext.BaseDirectory, "Models", TokenizerResource);
            var bytes = File.ReadAllBytes(spmModelPath);
            _spmModelStream = new MemoryStream(bytes);
            _tokenizer = SentencePieceTokenizer.Create(_spmModelStream);
        }

        public void Dispose()
        {
            _spmModelStream?.Dispose();
            _session?.Dispose();
        }

        public async Task<List<ClassificationResult>> ClassifyAsync(string inputText, string[] labels)
        {
            var labelPrompts = labels.Select(label => PromptTemplate(label)).ToList();

            var inputEmbedding = await GetEmbeddingAsync(inputText);
            var labelEmbeddings = await Task.WhenAll(labelPrompts.Select(GetEmbeddingAsync));

            var results = new List<ClassificationResult>();

            for (int i = 0; i < labels.Length; i++)
            {
                var similarity = CosineSimilarity(inputEmbedding, labelEmbeddings[i]);
                results.Add(new ClassificationResult(labels[i], similarity));
            }

            return results.OrderByDescending(r => r.Confidence).ToList();
        }

        private string PromptTemplate(string label)
        {
            return label.ToLower() switch
            {
                "interested" => "This person is interested.",
                "not interested" => "This person is not interested.",
                "do not call" => "This message asks not to be contacted.",
                _ => $"This text relates to '{label}'."
            };
        }

        private async Task<float[]> GetEmbeddingAsync(string text)
        {
            // Preprocess and tokenize (adjust for your model)
            var tokens = Tokenize(text);
            var inputIds = new DenseTensor<long>(new[] { 1, tokens.Length });
            for (int i = 0; i < tokens.Length; i++)
                inputIds[0, i] = tokens[i];

            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            // If your model requires attention_mask:
            NamedOnnxValue.CreateFromTensor("attention_mask", CreateAttentionMask(tokens.Length))
        };

            // Run ONNX inference asynchronously
            return await Task.Run(() =>
            {
                using var results = _session.Run(inputs);
                var embedding = results.First().AsEnumerable<float>().ToArray();
                return embedding;
            });
        }

        private long[] Tokenize(string text)
        {
            // Replace with real tokenizer based on the model used
            throw new NotImplementedException("Use a tokenizer consistent with your model (e.g., tokenizer.json from Hugging Face).");
        }

        private DenseTensor<long> CreateAttentionMask(int length)
        {
            var mask = new DenseTensor<long>(new[] { 1, length });
            for (int i = 0; i < length; i++) mask[0, i] = 1;
            return mask;
        }

        private float CosineSimilarity(float[] a, float[] b)
        {
            float dot = 0f, normA = 0f, normB = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            return dot / ((float)Math.Sqrt(normA) * (float)Math.Sqrt(normB) + 1e-8f);
        }
        private static byte[] StreamToByteArray(Stream stream)
        {
            if (stream is MemoryStream memoryStream)
            {
                return memoryStream.ToArray();
            }
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            return ms.ToArray();
        }

        
    }
}
