#include "air/core/Assert.h"
#include "air/renderer/RenderFeature.h"

namespace air::renderer {

void FeatureRegistry::Register(FeatureInfo info) {
    AIR_CHECK(!info.name.empty() && info.create, "feature registration needs a name and a factory");
    AIR_CHECK(!Find(info.name), "feature '{}' registered twice", info.name);
    mFeatures.push_back(std::move(info));
}

const FeatureInfo* FeatureRegistry::Find(std::string_view name) const {
    for (const FeatureInfo& info : mFeatures) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

std::unique_ptr<RenderFeature> FeatureRegistry::Create(std::string_view name) const {
    const FeatureInfo* info = Find(name);
    return info ? info->create() : nullptr;
}

} // namespace air::renderer
