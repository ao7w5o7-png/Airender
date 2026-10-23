#include "TestFramework.h"

#include "air/core/Image.h"
#include "air/core/Json.h"

#include <filesystem>

using namespace air;

TEST(Json_DumpIsDeterministicAndEscaped)
{
    json::Value v;
    v["b"] = 2;
    v["a"] = "quote\" newline\n";
    v["list"].Push(true);
    v["list"].Push(1.5);
    v["list"].Push(nullptr);
    EXPECT(v.Dump(0) == R"({"a":"quote\" newline\n","b":2,"list":[true,1.5,null]})");
}

TEST(Image_PfmAndPpmRoundTrip)
{
    Image image(5, 3);
    for (u32 y = 0; y < 3; ++y)
    {
        for (u32 x = 0; x < 5; ++x)
        {
            float* p = image.At(x, y);
            p[0] = float(x) / 4.0f;
            p[1] = float(y) / 2.0f;
            p[2] = 0.25f;
            p[3] = 1.0f;
        }
    }
    const auto dir = std::filesystem::temp_directory_path();
    EXPECT(WritePfm(dir / "air_test.pfm", image));
    auto pfm = ReadPfm(dir / "air_test.pfm");
    EXPECT(pfm && CompareImages(image, *pfm, {.pixelThreshold = 0.0f, .maxRmse = 0.0f}).passed);

    EXPECT(WritePpm(dir / "air_test.ppm", image));
    auto ppm = ReadImage(dir / "air_test.ppm");
    // 8-bit sRGB quantisation: within a couple of codes.
    EXPECT(ppm && CompareImages(image, *ppm, {.pixelThreshold = 0.02f, .maxRmse = 0.01f}).passed);
}

TEST(Image_CompareReportsDifferences)
{
    Image a(4, 4), b(4, 4);
    b.At(1, 1)[0] = 1.0f;
    const auto result = CompareImages(a, b, {.pixelThreshold = 0.1f, .maxRmse = 1.0f, .maxDifferingFraction = 0.0f});
    EXPECT(!result.passed);
    EXPECT(result.differingPixels == 1);
    EXPECT(result.maxAbsDiff == 1.0);
    EXPECT(result.diff.At(1, 1)[0] == 1.0f);
    EXPECT(CompareImages(a, Image(2, 2)).sizeMismatch);
}
