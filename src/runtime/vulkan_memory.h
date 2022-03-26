#ifndef HALIDE_RUNTIME_VULKAN_MEMORY_H
#define HALIDE_RUNTIME_VULKAN_MEMORY_H

#include "internal/block_allocator.h"
#include "vulkan_context.h"
#include "vulkan_internal.h"

namespace Halide {
namespace Runtime {
namespace Internal {
namespace Vulkan {

// Enable external client to override Vulkan allocation callbacks (if they so desire)
WEAK ScopedSpinLock::AtomicFlag custom_allocation_callbacks_lock = 0;
static const VkAllocationCallbacks* custom_allocation_callbacks = nullptr; // nullptr => use Vulkan runtime implementation

// --

// Halide System allocator for host allocations
WEAK void *vk_system_malloc(void *user_context, size_t size) {
    return malloc(size);
}

WEAK void vk_system_free(void *user_context, void *ptr) {
    free(ptr);
}

// Vulkan host-side allocation 
WEAK void* vk_host_malloc(void *user_context, size_t size, size_t alignment, VkSystemAllocationScope scope, const VkAllocationCallbacks* callbacks) {
    if(callbacks) {
        return callbacks->pfnAllocation(user_context, size, alignment, scope);
    } else {
        return vk_system_malloc(user_context, size);
    }
}

WEAK void vk_host_free(void *user_context, void *ptr, const VkAllocationCallbacks* callbacks) {
    if(callbacks) {
        return callbacks->pfnFree(user_context, ptr);
    } else {
        return vk_system_free(user_context, ptr);
    }
}

WEAK SystemMemoryAllocatorFns system_allocator = {vk_system_malloc, vk_system_free};

// Vulkan Memory allocator for host-device allocations
class VulkanMemoryAllocator;
WEAK VulkanMemoryAllocator *memory_allocator = nullptr;

// Runtime configuration parameters to adjust the behaviour of the block allocator
struct VulkanMemoryConfig {
    size_t minimum_block_size = 32 * 1024 * 1024;  // 32MB
    size_t maximum_block_size = 0;                 //< zero means no contraint
    size_t maximum_block_count = 0;                //< zero means no constraint
};
WEAK VulkanMemoryConfig memory_allocator_config;

// --

/** Vulkan Memory Allocator class interface for managing large 
 * memory requests stored as contiguous blocks of memory, which 
 * are then sub-allocated into smaller regions of 
 * memory to avoid the excessive cost of vkAllocate and the limited
 * number of available allocation calls through the API. 
*/
class VulkanMemoryAllocator {
public:
    // disable copy constructors and assignment
    VulkanMemoryAllocator(const VulkanMemoryAllocator &) = delete;
    VulkanMemoryAllocator &operator=(const VulkanMemoryAllocator &) = delete;

    // disable non-factory constrction
    VulkanMemoryAllocator() = delete;
    ~VulkanMemoryAllocator() = delete;

    // Factory methods for creation / destruction
    static VulkanMemoryAllocator *create(void *user_context, const VulkanMemoryConfig &config, const SystemMemoryAllocatorFns &system_allocator);
    static void destroy(void *user_context, VulkanMemoryAllocator *allocator);

    // Public interface methods
    MemoryRegion *reserve(void *user_context, MemoryRequest &request);
    void reclaim(void *user_context, MemoryRegion *region);
    bool collect(void *user_context);  //< returns true if any blocks were removed
    void destroy(void *user_context);

    void *map(void *user_context, MemoryRegion *region);
    void unmap(void *user_context, MemoryRegion *region);

    void bind(void *user_context, VkDevice device, VkPhysicalDevice physical_device);
    void unbind(void *user_context);

    static const VulkanMemoryConfig &default_config();

    static void allocate_block(void *user_context, MemoryBlock *block);
    static void deallocate_block(void *user_context, MemoryBlock *block);

    static void allocate_region(void *user_context, MemoryRegion *region);
    static void deallocate_region(void *user_context, MemoryRegion *region);

    size_t bytes_allocated_for_blocks() const;
    size_t blocks_allocated() const;

    size_t bytes_allocated_for_regions() const;
    size_t regions_allocated() const;

private:
    static const uint32_t invalid_usage_flags = uint32_t(-1);
    static const uint32_t invalid_memory_type = uint32_t(VK_MAX_MEMORY_TYPES);

    // Initializes a new instance
    void initialize(void *user_context, const VulkanMemoryConfig &config, const SystemMemoryAllocatorFns &system_allocator);

    uint32_t select_memory_usage(void *user_context, MemoryProperties properties) const;

    uint32_t select_memory_type(void *user_context,
                                VkPhysicalDevice physical_device,
                                MemoryProperties properties,
                                uint32_t required_flags) const;

private:
    size_t block_byte_count = 0;
    size_t block_count = 0;
    size_t region_byte_count = 0;
    size_t region_count = 0;
    VulkanMemoryConfig config;
    VkDevice device = nullptr;
    VkPhysicalDevice physical_device = nullptr;
    BlockAllocator *block_allocator = nullptr;
    ScopedSpinLock::AtomicFlag spin_lock = 0;
};

VulkanMemoryAllocator *VulkanMemoryAllocator::create(void *user_context, const VulkanMemoryConfig &cfg, const SystemMemoryAllocatorFns &system_allocator) {
    halide_abort_if_false(user_context, system_allocator.allocate != nullptr);
    VulkanMemoryAllocator *result = reinterpret_cast<VulkanMemoryAllocator *>(
        system_allocator.allocate(user_context, sizeof(VulkanMemoryAllocator)));

    if (result == nullptr) {
        error(user_context) << "VulkanMemoryAllocator: Failed to create instance! Out of memory!\n";
        return nullptr;
    }

    result->initialize(user_context, cfg, system_allocator);
    return result;
}

void VulkanMemoryAllocator::destroy(void *user_context, VulkanMemoryAllocator *instance) {
    halide_abort_if_false(user_context, instance != nullptr);
    const BlockAllocator::MemoryAllocators &allocators = instance->block_allocator->current_allocators();
    instance->destroy(user_context);
    BlockAllocator::destroy(user_context, instance->block_allocator);
    halide_abort_if_false(user_context, allocators.system.deallocate != nullptr);
    allocators.system.deallocate(user_context, instance);
}

void VulkanMemoryAllocator::initialize(void *user_context, const VulkanMemoryConfig &cfg, const SystemMemoryAllocatorFns &system_allocator) {
    spin_lock = 0;
    config = cfg;
    device = nullptr;
    physical_device = nullptr;
    BlockAllocator::MemoryAllocators allocators;
    allocators.system = system_allocator;
    allocators.block = {VulkanMemoryAllocator::allocate_block, VulkanMemoryAllocator::deallocate_block};
    allocators.region = {VulkanMemoryAllocator::allocate_region, VulkanMemoryAllocator::deallocate_region};
    BlockAllocator::Config block_allocator_config = {0};
    block_allocator_config.maximum_block_count = cfg.maximum_block_count;
    block_allocator_config.maximum_block_size = cfg.maximum_block_size;
    block_allocator_config.minimum_block_size = cfg.minimum_block_size;
    block_allocator = BlockAllocator::create(user_context, block_allocator_config, allocators);
    halide_abort_if_false(user_context, block_allocator != nullptr);
}

MemoryRegion *VulkanMemoryAllocator::reserve(void *user_context, MemoryRequest &request) {
    debug(0) << "VulkanMemoryAllocator: Reserving memory ("
             << "user_context=" << user_context << " "
             << "block_allocator=" << (void *)(block_allocator) << " "
             << "request.size=" << (uint32_t)(request.size) << " "
             << "device=" << (void *)(device) << " "
             << "physical_device=" << (void *)(physical_device) << ") ...\n";

    halide_abort_if_false(user_context, device != nullptr);
    halide_abort_if_false(user_context, physical_device != nullptr);
    halide_abort_if_false(user_context, block_allocator != nullptr);
    //    ScopedSpinLock lock(&spin_lock);
    return block_allocator->reserve(this, request);
}

void *VulkanMemoryAllocator::map(void *user_context, MemoryRegion *region) {
    debug(0) << "VulkanMemoryAllocator: Mapping region ("
             << "user_context=" << user_context << " "
             << "region=" << (void *)(region) << " "
             << "device=" << (void *)(device) << " "
             << "physical_device=" << (void *)(physical_device) << ") ...\n";

    halide_abort_if_false(user_context, device != nullptr);
    halide_abort_if_false(user_context, physical_device != nullptr);
    halide_abort_if_false(user_context, block_allocator != nullptr);
    //    ScopedSpinLock lock(&spin_lock);
    RegionAllocator *region_allocator = RegionAllocator::find_allocator(user_context, region);
    BlockResource *block_resource = region_allocator->block_resource();
    VkDeviceMemory device_memory = reinterpret_cast<VkDeviceMemory>(block_resource->memory.handle);

    uint8_t *mapped_ptr = nullptr;
    VkResult result = vkMapMemory(device, device_memory, region->offset, region->size, 0, (void **)(&mapped_ptr));
    if (result != VK_SUCCESS) {
        error(user_context) << "VulkanMemoryAllocator: Mapping region failed! vkMapMemory returned error code: " << get_vulkan_error_name(result) << "\n";
        return nullptr;
    }

    return mapped_ptr;
}

void VulkanMemoryAllocator::unmap(void *user_context, MemoryRegion *region) {
    debug(0) << "VulkanMemoryAllocator: Unmapping region ("
             << "user_context=" << user_context << " "
             << "region=" << (void *)(region) << " "
             << "device=" << (void *)(device) << " "
             << "physical_device=" << (void *)(physical_device) << ") ...\n";

    halide_abort_if_false(user_context, device != nullptr);
    halide_abort_if_false(user_context, physical_device != nullptr);
    //    ScopedSpinLock lock(&spin_lock);
    RegionAllocator *region_allocator = RegionAllocator::find_allocator(user_context, region);
    BlockResource *block_resource = region_allocator->block_resource();
    VkDeviceMemory device_memory = reinterpret_cast<VkDeviceMemory>(block_resource->memory.handle);
    vkUnmapMemory(device, device_memory);
}

WEAK void VulkanMemoryAllocator::bind(void *user_context, VkDevice dev, VkPhysicalDevice physical_dev) {
    debug(0) << "VulkanMemoryAllocator: Binding context ("
             << "user_context=" << user_context << " "
             << "device=" << (void *)(dev) << " "
             << "physical_device=" << (void *)(physical_dev) << ") ...\n";

    //    ScopedSpinLock lock(&spin_lock);
    device = dev;
    physical_device = physical_dev;
}

WEAK void VulkanMemoryAllocator::unbind(void *user_context) {
    debug(0) << "VulkanMemoryAllocator: Unbinding context ("
             << "user_context=" << user_context << " "
             << "device=" << (void *)(device) << " "
             << "physical_device=" << (void *)(physical_device) << ") ...\n";

    //    ScopedSpinLock lock(&spin_lock);
    device = nullptr;
    physical_device = nullptr;
}

void VulkanMemoryAllocator::reclaim(void *user_context, MemoryRegion *region) {
    debug(0) << "VulkanMemoryAllocator: Reclaiming region ("
             << "user_context=" << user_context << " "
             << "region=" << (void *)(region) << ") ... \n";

    halide_abort_if_false(user_context, device != nullptr);
    halide_abort_if_false(user_context, physical_device != nullptr);
    //    ScopedSpinLock lock(&spin_lock);
    return block_allocator->reclaim(this, region);
}

bool VulkanMemoryAllocator::collect(void *user_context) {
    debug(0) << "VulkanMemoryAllocator: Collecting unused memory ("
             << "user_context=" << user_context << ") ... \n";

    halide_abort_if_false(user_context, device != nullptr);
    halide_abort_if_false(user_context, physical_device != nullptr);
    //    ScopedSpinLock lock(&spin_lock);
    return block_allocator->collect(this);
}

void VulkanMemoryAllocator::destroy(void *user_context) {
    debug(0) << "VulkanMemoryAllocator: Destroying allocator ("
             << "user_context=" << user_context << ") ... \n";

    halide_abort_if_false(user_context, device != nullptr);
    halide_abort_if_false(user_context, physical_device != nullptr);
    //    ScopedSpinLock lock(&spin_lock);
    block_allocator->destroy(this);
}

const VulkanMemoryConfig &
VulkanMemoryAllocator::default_config() {
    static VulkanMemoryConfig result;
    return result;
}

// --

void VulkanMemoryAllocator::allocate_block(void *user_context, MemoryBlock *block) {
    debug(0) << "VulkanMemoryAllocator: Allocating block ("
             << "user_context=" << user_context << " "
             << "block=" << (void *)(block) << ") ... \n";

    VulkanMemoryAllocator *instance = reinterpret_cast<VulkanMemoryAllocator *>(user_context);
    halide_abort_if_false(user_context, instance != nullptr);
    halide_abort_if_false(user_context, instance->device != nullptr);
    halide_abort_if_false(user_context, instance->physical_device != nullptr);
    halide_abort_if_false(user_context, block != nullptr);

    debug(0) << "VulkanMemoryAllocator: Allocating block ("
             << "size=" << (uint32_t)block->size << ", "
             << "dedicated=" << (block->dedicated ? "true" : "false") << " "
             << "usage=" << halide_memory_usage_name(block->properties.usage) << " "
             << "caching=" << halide_memory_caching_name(block->properties.caching) << " "
             << "visibility=" << halide_memory_visibility_name(block->properties.visibility) << ")\n";

    // Find an appropriate memory type given the flags
    uint32_t memory_type = instance->select_memory_type(user_context, instance->physical_device, block->properties, 0);
    if (memory_type == invalid_memory_type) {
        debug(0) << "VulkanMemoryAllocator: Unable to find appropriate memory type for device!\n";
        return;
    }

    // Allocate memory
    VkMemoryAllocateInfo alloc_info = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,  // struct type
        nullptr,                                 // struct extending this
        block->size,                             // size of allocation in bytes
        memory_type                              // memory type index from physical device
    };

    VkDeviceMemory device_memory = {0};
    VkResult result = vkAllocateMemory(instance->device, &alloc_info, nullptr, &device_memory);
    if (result != VK_SUCCESS) {
        debug(0) << "VulkanMemoryAllocator: Allocation failed! vkAllocateMemory returned: " << get_vulkan_error_name(result) << "\n";
        return;
    }

    block->handle = (void *)device_memory;
    instance->block_byte_count += block->size;
    instance->block_count++;
}

void VulkanMemoryAllocator::deallocate_block(void *user_context, MemoryBlock *block) {
    debug(0) << "VulkanMemoryAllocator: Deallocating block ("
             << "user_context=" << user_context << " "
             << "block=" << (void *)(block) << ") ... \n";

    VulkanMemoryAllocator *instance = reinterpret_cast<VulkanMemoryAllocator *>(user_context);
    halide_abort_if_false(user_context, instance != nullptr);
    halide_abort_if_false(user_context, instance->device != nullptr);
    halide_abort_if_false(user_context, instance->physical_device != nullptr);
    halide_abort_if_false(user_context, block != nullptr);

    debug(0) << "VulkanBlockAllocator: deallocating block ("
             << "size=" << (uint32_t)block->size << ", "
             << "dedicated=" << (block->dedicated ? "true" : "false") << " "
             << "usage=" << halide_memory_usage_name(block->properties.usage) << " "
             << "caching=" << halide_memory_caching_name(block->properties.caching) << " "
             << "visibility=" << halide_memory_visibility_name(block->properties.visibility) << ")\n";

    if (block->handle == nullptr) {
        debug(0) << "VulkanBlockAllocator: Unable to deallocate block! Invalid handle!\n";
        return;
    }

    VkDeviceMemory device_memory = reinterpret_cast<VkDeviceMemory>(block->handle);
    vkFreeMemory(instance->device, device_memory, nullptr);
    instance->block_byte_count -= block->size;
    instance->block_count--;
}

size_t VulkanMemoryAllocator::blocks_allocated() const {
    return block_count;
}

size_t VulkanMemoryAllocator::bytes_allocated_for_blocks() const {
    return block_byte_count;
}

uint32_t VulkanMemoryAllocator::select_memory_type(void *user_context,
                                                   VkPhysicalDevice physical_device,
                                                   MemoryProperties properties,
                                                   uint32_t required_flags) const {

    uint32_t want_flags = 0;  //< preferred memory flags for requested access type
    uint32_t need_flags = 0;  //< must have in order to enable requested access
    switch (properties.visibility) {
    case MemoryVisibility::HostOnly:
        want_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        break;
    case MemoryVisibility::DeviceOnly:
        need_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case MemoryVisibility::DeviceToHost:
        need_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        want_flags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case MemoryVisibility::HostToDevice:
        need_flags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        break;
    case MemoryVisibility::DefaultVisibility:
    case MemoryVisibility::InvalidVisibility:
    default:
        debug(0) << "VulkanMemoryAllocator: Unable to convert type! Invalid memory visibility request!\n\t"
                 << "visibility=" << halide_memory_visibility_name(properties.visibility) << "\n";
        return invalid_memory_type;
    };

    switch (properties.caching) {
    case MemoryCaching::CachedCoherent:
        if (need_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            want_flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        break;
    case MemoryCaching::UncachedCoherent:
        if (need_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            want_flags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        break;
    case MemoryCaching::Cached:
        if (need_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            want_flags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        }
        break;
    case MemoryCaching::Uncached:
    case MemoryCaching::DefaultCaching:
        break;
    case MemoryCaching::InvalidCaching:
    default:
        debug(0) << "VulkanMemoryAllocator: Unable to convert type! Invalid memory caching request!\n\t"
                 << "caching=" << halide_memory_caching_name(properties.caching) << "\n";
        return invalid_memory_type;
    };

    VkPhysicalDeviceMemoryProperties device_memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &device_memory_properties);

    uint32_t result = invalid_memory_type;
    for (uint32_t i = 0; i < device_memory_properties.memoryTypeCount; ++i) {

        // if required flags are given, see if the memory type matches the requirement
        if (required_flags) {
            if (((required_flags >> i) & 1) == 0) {
                continue;
            }
        }

        const VkMemoryPropertyFlags properties = device_memory_properties.memoryTypes[i].propertyFlags;
        if (need_flags) {
            if ((properties & need_flags) != need_flags) {
                continue;
            }
        }

        if (want_flags) {
            if ((properties & want_flags) != want_flags) {
                continue;
            }
        }

        result = i;
        break;
    }

    if (result == invalid_memory_type) {
        debug(0) << "VulkanBlockAllocator: Failed to find appropriate memory type for given properties:\n\t"
                 << "usage=" << halide_memory_usage_name(properties.usage) << " "
                 << "caching=" << halide_memory_caching_name(properties.caching) << " "
                 << "visibility=" << halide_memory_visibility_name(properties.visibility) << "\n";
        return invalid_memory_type;
    }

    return result;
}

// --

void VulkanMemoryAllocator::allocate_region(void *user_context, MemoryRegion *region) {
    debug(0) << "VulkanMemoryAllocator: Allocating region ("
             << "user_context=" << user_context << " "
             << "region=" << (void *)(region) << ") ... \n";

    VulkanMemoryAllocator *instance = reinterpret_cast<VulkanMemoryAllocator *>(user_context);
    halide_abort_if_false(user_context, instance != nullptr);
    halide_abort_if_false(user_context, instance->device != nullptr);
    halide_abort_if_false(user_context, instance->physical_device != nullptr);
    halide_abort_if_false(user_context, region != nullptr);

    debug(0) << "VulkanRegionAllocator: Allocating region ("
             << "size=" << (uint32_t)region->size << ", "
             << "offset=" << (uint32_t)region->offset << ", "
             << "dedicated=" << (region->dedicated ? "true" : "false") << " "
             << "usage=" << halide_memory_usage_name(region->properties.usage) << " "
             << "caching=" << halide_memory_caching_name(region->properties.caching) << " "
             << "visibility=" << halide_memory_visibility_name(region->properties.visibility) << ")\n";

    uint32_t usage_flags = instance->select_memory_usage(user_context, region->properties);

    VkBufferCreateInfo create_info = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,  // struct type
        nullptr,                               // struct extending this
        0,                                     // create flags
        region->size,                          // buffer size (in bytes)
        usage_flags,                           // buffer usage flags
        VK_SHARING_MODE_EXCLUSIVE,             // sharing mode
        0, nullptr};

    VkBuffer buffer = {0};
    VkResult result = vkCreateBuffer(instance->device, &create_info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        error(user_context) << "VulkanRegionAllocator: Failed to create buffer!\n\t"
                            << "vkCreateBuffer returned: " << get_vulkan_error_name(result) << "\n";
        return;
    }

    RegionAllocator *region_allocator = RegionAllocator::find_allocator(user_context, region);
    BlockResource *block_resource = region_allocator->block_resource();
    VkDeviceMemory device_memory = reinterpret_cast<VkDeviceMemory>(block_resource->memory.handle);

    // Finally, bind buffer to the device memory
    result = vkBindBufferMemory(instance->device, buffer, device_memory, region->offset);
    if (result != VK_SUCCESS) {
        error(user_context) << "VulkanRegionAllocator: Failed to bind buffer!\n\t"
                            << "vkBindBufferMemory returned: " << get_vulkan_error_name(result) << "\n";
        return;
    }

    region->handle = (void *)buffer;
    instance->region_byte_count += region->size;
    instance->region_count++;
}

void VulkanMemoryAllocator::deallocate_region(void *user_context, MemoryRegion *region) {
    debug(0) << "VulkanMemoryAllocator: Deallocating region ("
             << "user_context=" << user_context << " "
             << "region=" << (void *)(region) << ") ... \n";

    VulkanMemoryAllocator *instance = reinterpret_cast<VulkanMemoryAllocator *>(user_context);
    halide_abort_if_false(user_context, instance != nullptr);
    halide_abort_if_false(user_context, instance->device != nullptr);
    halide_abort_if_false(user_context, instance->physical_device != nullptr);
    halide_abort_if_false(user_context, region != nullptr);
    debug(0) << "VulkanRegionAllocator: Deallocating region ("
             << "size=" << (uint32_t)region->size << ", "
             << "offset=" << (uint32_t)region->offset << ", "
             << "dedicated=" << (region->dedicated ? "true" : "false") << " "
             << "usage=" << halide_memory_usage_name(region->properties.usage) << " "
             << "caching=" << halide_memory_caching_name(region->properties.caching) << " "
             << "visibility=" << halide_memory_visibility_name(region->properties.visibility) << ")\n";

    if (region->handle == nullptr) {
        debug(0) << "VulkanRegionAllocator: Unable to deallocate region! Invalid handle!\n";
        return;
    }

    VkBuffer buffer = (VkBuffer)region->handle;
    vkDestroyBuffer(instance->device, buffer, nullptr);

    region->handle = nullptr;
    instance->region_byte_count -= region->size;
    instance->region_count--;
}

size_t VulkanMemoryAllocator::regions_allocated() const {
    return region_count;
}

size_t VulkanMemoryAllocator::bytes_allocated_for_regions() const {
    return region_byte_count;
}

uint32_t VulkanMemoryAllocator::select_memory_usage(void *user_context, MemoryProperties properties) const {
    uint32_t result = 0;
    switch (properties.usage) {
    case MemoryUsage::DynamicStorage:
    case MemoryUsage::StaticStorage:
        result |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        break;
    case MemoryUsage::TransferSrc:
        result |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        break;
    case MemoryUsage::TransferDst:
        result |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        break;
    case MemoryUsage::TransferSrcDst:
        result |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        break;
    case MemoryUsage::DefaultUsage:
    case MemoryUsage::InvalidUsage:
    default:
        error(user_context) << "VulkanRegionAllocator: Unable to convert type! Invalid memory usage request!\n\t"
                            << "usage=" << halide_memory_usage_name(properties.usage) << "\n";
        return invalid_usage_flags;
    };

    if (result == invalid_usage_flags) {
        error(user_context) << "VulkanRegionAllocator: Failed to find appropriate memory usage for given properties:\n\t"
                            << "usage=" << halide_memory_usage_name(properties.usage) << " "
                            << "caching=" << halide_memory_caching_name(properties.caching) << " "
                            << "visibility=" << halide_memory_visibility_name(properties.visibility) << "\n";
        return invalid_usage_flags;
    }

    return result;
}

// --

WEAK int vk_create_memory_allocator(void *user_context, const VkAllocationCallbacks* alloc_callbacks) {
    if (memory_allocator != nullptr) {
        return halide_error_code_success;
    }

    if(alloc_callbacks) {
        // TODO
    }

    memory_allocator = VulkanMemoryAllocator::create(user_context, memory_allocator_config, system_allocator);
    if (memory_allocator == nullptr) {
        return halide_error_code_out_of_memory;
    }
    return halide_error_code_success;
}

WEAK int vk_destroy_memory_allocator(void *user_context, const VkAllocationCallbacks* callbacks) {
    if (memory_allocator == nullptr) {
        return halide_error_code_success;
    }

    if(callbacks) {
        // TODO
    }

    VulkanMemoryAllocator::destroy(user_context, memory_allocator);
    memory_allocator = nullptr;
    return halide_error_code_success;
}

// --

}  // namespace Vulkan
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide


extern "C" {

WEAK void halide_vulkan_set_allocation_callbacks(const VkAllocationCallbacks* callbacks) {
    ScopedSpinLock lock(&custom_allocation_callbacks_lock);
    custom_allocation_callbacks = callbacks;
}

WEAK const VkAllocationCallbacks* halide_vulkan_get_allocation_callbacks(void *user_context) {
    ScopedSpinLock lock(&custom_allocation_callbacks_lock);
    return custom_allocation_callbacks;
}

}

#endif  // HALIDE_RUNTIME_VULKAN_MEMORY_H