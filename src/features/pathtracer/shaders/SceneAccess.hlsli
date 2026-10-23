// The integrator's scene hooks (Integrator.hlsli) over the bindless scene buffers, and inline shadow
// rays: shared by the trace kernel and the ReSTIR passes. The including shader declares `g`, its push
// constants, with the fields tlas, instances, materials, lights, instanceLights, lightTree and bsdfLut,
// and supplies TraceClosest itself (closest-hit shaders or an inline query).
#ifndef AIR_PT_SCENE_ACCESS_HLSLI
#define AIR_PT_SCENE_ACCESS_HLSLI

float BsdfLutLoad(uint index)
{
    StructuredBuffer<float> lut = ResourceDescriptorHeap[g.bsdfLut];
    return lut[index];
}

InstanceData LoadInstance(uint instance)
{
    StructuredBuffer<InstanceData> instances = ResourceDescriptorHeap[g.instances];
    return instances[instance];
}

uint LoadInstanceLight(uint instance)
{
    StructuredBuffer<uint> instanceLights = ResourceDescriptorHeap[g.instanceLights];
    return instanceLights[instance];
}

uint LoadIndex(InstanceData instance, uint index)
{
    StructuredBuffer<uint> indices = ResourceDescriptorHeap[NonUniformResourceIndex(instance.indexBuffer)];
    return indices[index];
}

Vertex LoadVertex(InstanceData instance, uint index)
{
    StructuredBuffer<Vertex> vertices = ResourceDescriptorHeap[NonUniformResourceIndex(instance.vertexBuffer)];
    return vertices[index];
}

MaterialData LoadMaterial(uint material)
{
    StructuredBuffer<MaterialData> materials = ResourceDescriptorHeap[g.materials];
    return materials[material];
}

LightTriangle LoadLight(uint light)
{
    StructuredBuffer<LightTriangle> lights = ResourceDescriptorHeap[g.lights];
    return lights[light];
}

LightTreeNode LoadLightNode(uint node)
{
    StructuredBuffer<LightTreeNode> tree = ResourceDescriptorHeap[g.lightTree];
    return tree[node];
}

bool TraceVisible(float3 origin, float3 direction, float tMax)
{
    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[g.tlas];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0f;
    ray.TMax = tMax;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();
    return query.CommittedStatus() == COMMITTED_NOTHING;
}

#endif
