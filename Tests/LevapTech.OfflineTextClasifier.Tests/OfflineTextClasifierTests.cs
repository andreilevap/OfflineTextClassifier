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
    }
}
