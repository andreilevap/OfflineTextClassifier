namespace LevapTech.OfflineTextClassifier.Tests
{
    public class OfflineTextClasifierTests
    {
        [Fact]
        public void Constructor_Default_DoesNotThrow()
        {
            var classifier = new OfflineTextClasifier();
            Assert.False(classifier.IsLightweight);
        }

        [Fact]
        public void Constructor_Lightweight_SetsIsLightweight()
        {
            var classifier = new OfflineTextClasifier(isLightweight: true);
            Assert.True(classifier.IsLightweight);
        }

        [Fact]
        public async Task ClassifyAsync_BasicInput_ReturnsResults()
        {
            var classifier = new OfflineTextClasifier();
            var labels = new[] { "politics", "football", "art", "travel" };
            var results = await classifier.ClassifyAsync("Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU", labels);

            Assert.NotNull(results);
            Assert.Equal(labels.Length, results.Count);

            Assert.All(results, r => Assert.Contains(r.Label, labels));
        }

    }
}
