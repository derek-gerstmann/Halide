#include "HalideRuntimeVulkan.h"

#include "device_buffer_utils.h"
#include "device_interface.h"
#include "runtime_internal.h"
#include "vulkan_context.h"
#include "vulkan_extensions.h"
#include "vulkan_internal.h"
#include "vulkan_memory.h"
#include "vulkan_compiler.h"

using namespace Halide::Runtime::Internal::Vulkan;

// --------------------------------------------------------------------------

extern "C" {

// --------------------------------------------------------------------------

// The default implementation of halide_acquire_vulkan_context uses
// the global pointers above, and serializes access with a spin lock.
// Overriding implementations of acquire/release must implement the
// following behavior:

//  - halide_acquire_vulkan_context should always store a valid
//   instance/device/queue in the corresponding out parameters,
//   or return an error code.
// - A call to halide_acquire_vulkan_context is followed by a matching
//   call to halide_release_vulkan_context. halide_acquire_vulkan_context
//   should block while a previous call (if any) has not yet been
//   released via halide_release_vulkan_context.
WEAK int halide_vulkan_acquire_context(void *user_context, 
                                       halide_vulkan_memory_allocator **allocator,
                                       VkInstance *instance,
                                       VkDevice *device, VkQueue *queue, 
                                       VkPhysicalDevice *physical_device,
                                       uint32_t *queue_family_index, 
                                       bool create) {

    halide_abort_if_false(user_context, instance != nullptr);
    halide_abort_if_false(user_context, device != nullptr);
    halide_abort_if_false(user_context, queue != nullptr);
    halide_abort_if_false(user_context, &thread_lock != nullptr);
    while (__atomic_test_and_set(&thread_lock, __ATOMIC_ACQUIRE)) {}

    // If the context has not been initialized, initialize it now.
    halide_abort_if_false(user_context, &cached_instance != nullptr);
    halide_abort_if_false(user_context, &cached_device != nullptr);
    halide_abort_if_false(user_context, &cached_queue != nullptr);
    halide_abort_if_false(user_context, &cached_physical_device != nullptr);
    if ((cached_instance == nullptr) && create) {
        int result = vk_create_context(user_context, 
            reinterpret_cast<VulkanMemoryAllocator**>(&cached_allocator), 
            &cached_instance, 
            &cached_device, 
            &cached_queue, 
            &cached_physical_device, 
            &cached_queue_family_index
        );
        if (result != halide_error_code_success) {
            __atomic_clear(&thread_lock, __ATOMIC_RELEASE);
            return result;
        }
    }

    *allocator = cached_allocator;
    *instance = cached_instance;
    *device = cached_device;
    *queue = cached_queue;
    *physical_device = cached_physical_device;
    *queue_family_index = cached_queue_family_index;
    return 0;
}

WEAK int halide_vulkan_release_context(void *user_context, VkInstance instance, VkDevice device, VkQueue queue) {
    __atomic_clear(&thread_lock, __ATOMIC_RELEASE);
    return 0;
}

WEAK int halide_vulkan_device_free(void *user_context, halide_buffer_t *halide_buffer) {
    // halide_vulkan_device_free, at present, can be exposed to clients and they
    // should be allowed to call halide_vulkan_device_free on any halide_buffer_t
    // including ones that have never been used with a GPU.
    if (halide_buffer->device == 0) {
        return 0;
    }

    VulkanContext ctx(user_context);

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    // get the allocated region for the device
    MemoryRegion* device_region = reinterpret_cast<MemoryRegion*>(halide_buffer->device);
    if(ctx.allocator && device_region && device_region->handle) {
        ctx.allocator->reclaim(user_context, device_region);
    }
    halide_buffer->device = 0;
    halide_buffer->device_interface->impl->release_module();
    halide_buffer->device_interface = nullptr;

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return 0;
}

WEAK int halide_vulkan_initialize_kernels(void *user_context, void **state_ptr, const char *src, int size) {
    debug(user_context)
        << "Vulkan: halide_vulkan_init_kernels (user_context: " << user_context
        << ", state_ptr: " << state_ptr
        << ", program: " << (void *)src
        << ", size: " << size << "\n";

    VulkanContext ctx(user_context);
    if (ctx.error != VK_SUCCESS) {
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    debug(user_context) << "halide_vulkan_initialize_kernels got compilation_cache mutex.\n";
    VkShaderModule* shader_module = nullptr;
    if (!compilation_cache.kernel_state_setup(user_context, state_ptr, ctx.device, shader_module,
                                              Halide::Runtime::Internal::Vulkan::vk_compile_shader_module, 
                                              user_context, ctx.allocator, src, size)) {
        return halide_error_code_generic_error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return 0;
}

WEAK void halide_vulkan_finalize_kernels(void *user_context, void *state_ptr) {
    debug(user_context)
        << "Vulkan: halide_vulkan_finalize_kernels (user_context: " << user_context
        << ", state_ptr: " << state_ptr << "\n";
    VulkanContext ctx(user_context);
    if (ctx.error == VK_SUCCESS) {
        compilation_cache.release_hold(user_context, ctx.device, state_ptr);
    }
}

// Used to generate correct timings when tracing
WEAK int halide_vulkan_device_sync(void *user_context, halide_buffer_t *) {
    debug(user_context) << "Vulkan: halide_vulkan_device_sync (user_context: " << user_context << ")\n";

    VulkanContext ctx(user_context);
    halide_debug_assert(user_context, ctx.error == VK_SUCCESS);

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return VK_SUCCESS;
}

WEAK int halide_vulkan_device_release(void *user_context) {
    debug(user_context)
        << "Vulkan: halide_vulkan_device_release (user_context: " << user_context << ")\n";

    VkInstance instance;
    VkDevice device;
    VkQueue queue;
    VulkanMemoryAllocator* allocator;
    VkPhysicalDevice physical_device;
    uint32_t _throwaway;
    
    int acquire_status = halide_vulkan_acquire_context(user_context, 
        reinterpret_cast<halide_vulkan_memory_allocator**>(&allocator), 
        &instance, &device, &queue, &physical_device, &_throwaway, false
    );
    halide_debug_assert(user_context, acquire_status == VK_SUCCESS);
    (void)acquire_status;
    if (instance != nullptr) {

        vkQueueWaitIdle(queue);
        vk_destroy_shader_modules(user_context, allocator);      
        vk_destroy_memory_allocator(user_context, allocator);

        if (device == cached_device) {
            cached_device = nullptr;
            cached_physical_device = nullptr;
            cached_queue = nullptr;
        }
        vkDestroyDevice(device, nullptr);

        if (instance == cached_instance) {
            cached_instance = nullptr;
        }
        vkDestroyInstance(instance, nullptr);
        halide_vulkan_release_context(user_context, instance, device, queue);
    }

    return 0;
}

namespace {

VkResult vk_create_command_pool(VkDevice device, uint32_t queue_index, const VkAllocationCallbacks *callbacks, VkCommandPool *command_pool) {

    VkCommandPoolCreateInfo command_pool_info =
        {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,  // struct type
            nullptr,                                     // pointer to struct extending this
            0,                                           // flags.  may consider VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
            queue_index                                  // queue family index corresponding to the compute command queue
        };
    return vkCreateCommandPool(device, &command_pool_info, callbacks, command_pool);
}

VkResult vk_create_command_buffer(VkDevice device, VkCommandPool pool, VkCommandBuffer *command_buffer) {

    VkCommandBufferAllocateInfo command_buffer_info =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,  // struct type
            nullptr,                                         // pointer to struct extending this
            pool,                                            // command pool for allocation
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,                 // command buffer level
            1                                                // number to allocate
        };

    return vkAllocateCommandBuffers(device, &command_buffer_info, command_buffer);
}

}  // anonymous namespace

WEAK int halide_vulkan_device_malloc(void *user_context, halide_buffer_t *buf) {
    debug(user_context)
        << "halide_vulkan_device_malloc (user_context: " << user_context
        << ", buf: " << buf << ")\n";

    VulkanContext ctx(user_context);
    if (ctx.error != VK_SUCCESS) {
        return -1;
    }

    size_t size = buf->size_in_bytes();
    halide_debug_assert(user_context, size != 0);
    if (buf->device) {
        return 0;
    }

    for (int i = 0; i < buf->dimensions; i++) {
        halide_debug_assert(user_context, buf->dim[i].stride >= 0);
    }

    debug(user_context) << "    allocating " << *buf << "\n";

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    // request uncached device only memory
    MemoryRequest request = {0};
    request.size = size;
    request.properties.usage = MemoryUsage::TransferSrcDst;
    request.properties.caching = MemoryCaching::Uncached;
    request.properties.visibility = MemoryVisibility::DeviceOnly;

    // allocate a new region
    MemoryRegion *device_region = ctx.allocator->reserve(user_context, request);
    if ((device_region == nullptr) || (device_region->handle == nullptr)) {
        error(user_context) << "Vulkan: Failed to allocate device memory!\n";
        return -1;
    }

    buf->device = (uint64_t)device_region;
    buf->device_interface = &vulkan_device_interface;
    buf->device_interface->impl->use_module();

    debug(user_context)
        << "    Allocated device buffer " << (void *)buf->device
        << " for buffer " << buf << "\n";

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return 0;
}

namespace {

WEAK int do_multidimensional_copy(void *user_context, const VulkanContext &ctx,
                                  const device_copy &c,
                                  uint64_t off, int d, bool d_to_h) {
    if (d > MAX_COPY_DIMS) {
        error(user_context) << "Buffer has too many dimensions to copy to/from GPU\n";
        return -1;
    } else if (d == 0) {
        void vkCmdCopyBuffer(
            VkCommandBuffer commandBuffer,
            VkBuffer srcBuffer,
            VkBuffer dstBuffer,
            uint32_t regionCount,
            const VkBufferCopy *pRegions);

    } else if (d == 2) {
    } else {
        for (int i = 0; i < (int)c.extent[d - 1]; i++) {
            int err = do_multidimensional_copy(user_context, ctx, c, off, d - 1, d_to_h);
            off += c.src_stride_bytes[d - 1];
            if (err) {
                return err;
            }
        }
    }
    return 0;
}
}  // namespace

WEAK int halide_vulkan_copy_to_device(void *user_context, halide_buffer_t *halide_buffer) {
    int err = halide_vulkan_device_malloc(user_context, halide_buffer);
    if (err) {
        return err;
    }

    debug(user_context)
        << "Vulkan: halide_vulkan_copy_to_device (user_context: " << user_context
        << ", halide_buffer: " << halide_buffer << ")\n";

    // Acquire the context so we can use the command queue.
    VulkanContext ctx(user_context);
    if (ctx.error != VK_SUCCESS) {
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    halide_abort_if_false(user_context, halide_buffer->host && halide_buffer->device);

    device_copy copy_helper = make_host_to_device_copy(halide_buffer);

    // We construct a staging buffer to copy into from host memory.  Then,
    // we use vkCmdCopyBuffer() to copy from the staging buffer into the
    // the actual device memory.
    MemoryRequest request = {0};
    request.size = halide_buffer->size_in_bytes();
    request.properties.usage = MemoryUsage::TransferSrc;
    request.properties.caching = MemoryCaching::UncachedCoherent;
    request.properties.visibility = MemoryVisibility::HostToDevice;

    // allocate a new region
    MemoryRegion *staging_region = ctx.allocator->reserve(user_context, request);
    if ((staging_region == nullptr) || (staging_region->handle == nullptr)) {
        error(user_context) << "Vulkan: Failed to allocate device memory!\n";
        return -1;
    }

    // map the region to a host ptr
    uint8_t *stage_host_ptr = (uint8_t *)ctx.allocator->map(user_context, staging_region);
    if (stage_host_ptr == nullptr) {
        error(user_context) << "Vulkan: Failed to map host pointer to device memory!\n";
        return halide_error_code_internal_error;
    }

    // copy to the (host-visible/coherent) staging buffer
    copy_helper.dst = (uint64_t)(stage_host_ptr);
    copy_memory(copy_helper, user_context);

   // retrieve the buffer from the region
    VkBuffer* staging_buffer = reinterpret_cast<VkBuffer*>(staging_region->handle);
    if (staging_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve staging buffer for device memory!\n";
        return halide_error_code_internal_error;
    }
    
    // unmap the pointer
    ctx.allocator->unmap(user_context, staging_region);

    // TODO: only copy the regions that should be copied
    VkBufferCopy staging_copy = {
        0,                              // srcOffset
        0,                              // dstOffset
        halide_buffer->size_in_bytes()  // size
    };

    // create a command buffer
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkResult result = vk_create_command_pool(ctx.device, ctx.queue_family_index, ctx.allocation_callbacks(), &command_pool);
    if (result != VK_SUCCESS) {
        debug(user_context) << "Vulkan: vkCreateCommandPool returned: " << vk_get_error_name(result) << "\n";
        return result;
    }

    result = vk_create_command_buffer(ctx.device, command_pool, &command_buffer);
    if (result != VK_SUCCESS) {
        debug(user_context) << "Vulkan: vkCreateCommandBuffer returned: " << vk_get_error_name(result) << "\n";
        return result;
    }

    // begin the command buffer
    VkCommandBufferBeginInfo command_buffer_begin_info =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // struct type
            nullptr,                                      // pointer to struct extending this
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
            nullptr                                       // pointer to parent command buffer
        };

    result = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
    if (result != VK_SUCCESS) {
        debug(user_context) << "vkBeginCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    // get the allocated region for the device
    MemoryRegion* device_region = reinterpret_cast<MemoryRegion*>(halide_buffer->device);

    // retrieve the buffer from the region
    VkBuffer* device_buffer = reinterpret_cast<VkBuffer*>(device_region->handle);
    if (device_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve buffer for device memory!\n";
        return halide_error_code_internal_error;
    }

    // enqueue the copy operation
    vkCmdCopyBuffer(command_buffer, *staging_buffer, *device_buffer, 1, &staging_copy);

    // end the command buffer
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        debug(user_context) << "vkEndCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 13. Submit the command buffer to our command queue
    VkSubmitInfo submit_info =
        {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,  // struct type
            nullptr,                        // pointer to struct extending this
            0,                              // wait semaphore count
            nullptr,                        // semaphores
            nullptr,                        // pipeline stages where semaphore waits occur
            1,                              // how many command buffers to execute
            &command_buffer,                // the command buffers
            0,                              // number of semaphores to signal
            nullptr                         // the semaphores to signal
        };

    result = vkQueueSubmit(ctx.queue, 1, &submit_info, 0);
    if (result != VK_SUCCESS) {
        debug(user_context) << "vkQueueSubmit returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 14. Wait until the queue is done with the command buffer
    result = vkQueueWaitIdle(ctx.queue);
    if (result != VK_SUCCESS) {
        debug(user_context) << "vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 15. Reclaim the staging buffer
    ctx.allocator->reclaim(user_context, staging_region);

    //do_multidimensional_copy(user_context, ctx, c, 0, buf->dimensions, false);

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return 0;
}

WEAK int halide_vulkan_copy_to_host(void *user_context, halide_buffer_t *halide_buffer) {

#ifdef DEBUG_RUNTIME
    debug(user_context)
        << "Vulkan: halide_copy_to_host (user_context: " << user_context
        << ", halide_buffer: " << halide_buffer << ")\n";
#endif

    // Acquire the context so we can use the command queue. This also avoids multiple
    // redundant calls to clEnqueueReadBuffer when multiple threads are trying to copy
    // the same buffer.
    VulkanContext ctx(user_context);
    if (ctx.error != VK_SUCCESS) {
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    halide_abort_if_false(user_context, halide_buffer->host && halide_buffer->device);

    device_copy copy_helper = make_device_to_host_copy(halide_buffer);

    //do_multidimensional_copy(user_context, ctx, c, 0, buf->dimensions, true);

    // This is the inverse of copy_to_device: we create a staging buffer, copy into
    // it, map it so the host can see it, then copy into the host buffer

    MemoryRequest request = {0};
    request.size = halide_buffer->size_in_bytes();
    request.properties.usage = MemoryUsage::TransferDst;
    request.properties.caching = MemoryCaching::UncachedCoherent;
    request.properties.visibility = MemoryVisibility::DeviceToHost;

    // allocate a new region for staging the transfer
    MemoryRegion *staging_region = ctx.allocator->reserve(user_context, request);
    if ((staging_region == nullptr) || (staging_region->handle == nullptr)) {
        error(user_context) << "Vulkan: Failed to allocate device memory!\n";
        return -1;
    }

    // retrieve the buffer from the region
    VkBuffer* staging_buffer = reinterpret_cast<VkBuffer*>(staging_region->handle);
    if (staging_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve staging buffer for device memory!\n";
        return halide_error_code_internal_error;
    }

    // TODO: only copy the regions that should be copied
    VkBufferCopy staging_copy = {
        0,                              // srcOffset
        0,                              // dstOffset
        halide_buffer->size_in_bytes()  // size
    };

    // create a command buffer
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkResult result = vk_create_command_pool(ctx.device, ctx.queue_family_index, ctx.allocation_callbacks(), &command_pool);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkCreateCommandPool returned: " << vk_get_error_name(result) << "\n";
        return -1;
    }

    result = vk_create_command_buffer(ctx.device, command_pool, &command_buffer);

    // begin the command buffer
    VkCommandBufferBeginInfo command_buffer_begin_info =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // struct type
            nullptr,                                      // pointer to struct extending this
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
            nullptr                                       // pointer to parent command buffer
        };

    result = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
    if (result != VK_SUCCESS) {
        error(user_context) << "vkBeginCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    // get the allocated region for the device
    MemoryRegion* device_region = reinterpret_cast<MemoryRegion*>(halide_buffer->device);

    // retrieve the buffer from the region
    VkBuffer* device_buffer = reinterpret_cast<VkBuffer*>(device_region->handle);
    if (device_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve buffer for device memory!\n";
        return halide_error_code_internal_error;
    }
        
    // enqueue the copy operation
    vkCmdCopyBuffer(command_buffer, *device_buffer, *staging_buffer, 1, &staging_copy);

    // end the command buffer
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        error(user_context) << "vkEndCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 13. Submit the command buffer to our command queue
    VkSubmitInfo submit_info =
        {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,  // struct type
            nullptr,                        // pointer to struct extending this
            0,                              // wait semaphore count
            nullptr,                        // semaphores
            nullptr,                        // pipeline stages where semaphore waits occur
            1,                              // how many command buffers to execute
            &command_buffer,                // the command buffers
            0,                              // number of semaphores to signal
            nullptr                         // the semaphores to signal
        };

    result = vkQueueSubmit(ctx.queue, 1, &submit_info, 0);
    if (result != VK_SUCCESS) {
        error(user_context) << "vkQueueSubmit returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 14. Wait until the queue is done with the command buffer
    result = vkQueueWaitIdle(ctx.queue);
    if (result != VK_SUCCESS) {
        error(user_context) << "vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    // map the staging region to a host ptr
    uint8_t *stage_host_ptr = (uint8_t *)ctx.allocator->map(user_context, staging_region);
    if (stage_host_ptr == nullptr) {
        error(user_context) << "Vulkan: Failed to map host pointer to device memory!\n";
        return halide_error_code_internal_error;
    }

    // copy to the (host-visible/coherent) staging buffer
    copy_helper.src = (uint64_t)(stage_host_ptr);
    copy_memory(copy_helper, user_context);

    // unmap the pointer and reclaim the staging region
    ctx.allocator->unmap(user_context, staging_region);
    ctx.allocator->reclaim(user_context, staging_region);

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return 0;
}

WEAK uint32_t vk_count_bindings_for_descriptor_set(void* user_context,                                          
                                                   size_t arg_sizes[],
                                                   void *args[],
                                                   int8_t arg_is_buffer[]){
    int i = 0;
    uint32_t num_bindings = 1; // first binding is for passing scalar parameters in a buffer
    while (arg_sizes[i] > 0) {
        if (arg_is_buffer[i]) {
            num_bindings++;
        }
        i++;
    }
    return num_bindings;
}

WEAK size_t vk_estimate_scalar_uniform_buffer_size(void* user_context,                                          
                                                   size_t arg_sizes[],
                                                   void *args[],
                                                   int8_t arg_is_buffer[]){
    int i = 0;
    int scalar_uniform_buffer_size = 0;
    while (arg_sizes[i] > 0) {
        if (!arg_is_buffer[i]) {
            scalar_uniform_buffer_size += arg_sizes[i];
        }
        i++;
    }
    return scalar_uniform_buffer_size;
}

WEAK MemoryRegion* vk_create_scalar_uniform_buffer(void* user_context, 
                                                   VulkanMemoryAllocator* allocator,
                                                   size_t arg_sizes[],
                                                   void *args[],
                                                   int8_t arg_is_buffer[]){

    debug(user_context)
        << "Vulkan: vk_create_scalar_uniform_buffer (user_context: " << user_context << ", "
        << "allocator: " << (void*)allocator << ")\n";

    size_t scalar_buffer_size = vk_estimate_scalar_uniform_buffer_size(user_context, 
        arg_sizes, args, arg_is_buffer);

    MemoryRequest request = {0};
    request.size = scalar_buffer_size;
    request.properties.usage = MemoryUsage::UniformStorage;
    request.properties.caching = MemoryCaching::UncachedCoherent;
    request.properties.visibility = MemoryVisibility::HostToDevice;

    // allocate a new region
    MemoryRegion *region = allocator->reserve(user_context, request);
    if ((region == nullptr) || (region->handle == nullptr)) {
        error(user_context) << "Vulkan: Failed to allocate device memory!\n";
        return nullptr;
    }

    // map the region to a host ptr
    uint8_t * scalar_buffer_host_ptr = (uint8_t *)allocator->map(user_context, region);
    if (scalar_buffer_host_ptr == nullptr) {
        error(user_context) << "Vulkan: Failed to map host pointer to device memory!\n";
        return nullptr;
    }

    // copy to the (host-visible/coherent) scalar uniform buffer
    size_t scalar_arg_offset = 0;
    debug(user_context) << "Parameter: (passed in vs value after copy)\n";
    for (size_t i = 0; arg_sizes[i] > 0; i++) {
        if (!arg_is_buffer[i]) {
            memcpy(scalar_buffer_host_ptr + scalar_arg_offset, args[i], arg_sizes[i]);
            debug(user_context) << *((int32_t *)(scalar_buffer_host_ptr + scalar_arg_offset));
            debug(user_context) << "   " << *((int32_t *)(args[i])) << "\n";
            scalar_arg_offset += arg_sizes[i];
        }
    }

    // unmap the pointer to the buffer for the region
    allocator->unmap(user_context, region);

    // return the allocated region for the uniform buffer
    return region;
}

WEAK void vk_destroy_scalar_uniform_buffer(void* user_context, VulkanMemoryAllocator* allocator,
                                           MemoryRegion* scalar_args_region){

    debug(user_context)
        << "Vulkan: vk_destroy_scalar_uniform_buffer (user_context: " << user_context << ", "
        << "allocator: " << (void*)allocator << ", "
        << "scalar_args_region: " << (void*)scalar_args_region << ")\n";

    if(!scalar_args_region) { return; }
    allocator->reclaim(user_context, scalar_args_region);
}

WEAK VkResult vk_create_descriptor_set_layout(void* user_context, 
                                         VkDevice device,
                                         size_t arg_sizes[],
                                         void *args[],
                                         int8_t arg_is_buffer[],
                                         VkDescriptorSetLayout* layout) {

    debug(user_context)
        << "Vulkan: vk_create_descriptor_set_layout (user_context: " << user_context << ", "
        << "device: " << (void*)device << ", "
        << "layout: " << (void*)layout << "\n";

    // The first binding is used for scalar parameters
    uint32_t num_bindings = vk_count_bindings_for_descriptor_set(user_context, arg_sizes, args, arg_is_buffer);
    
    BlockStorage::Config layout_config;
    layout_config.entry_size = sizeof(VkDescriptorSetLayoutBinding);
    layout_config.minimum_capacity = num_bindings;

    BlockStorage layout_bindings(user_context, layout_config);

    // First binding is reserved for passing scalar parameters as a uniform buffer
    VkDescriptorSetLayoutBinding scalar_uniform_layout = {
        0,                                  // binding index
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // descriptor type
        1,                                  // descriptor count
        VK_SHADER_STAGE_COMPUTE_BIT,        // stage flags
        0                                   // immutable samplers
    };
    layout_bindings.append(user_context, &scalar_uniform_layout);

    // Add all other bindings for buffer data
    int i = 0;
    while (arg_sizes[i] > 0) {
        if (arg_is_buffer[i]) {
            // TODO: I don't quite understand why STORAGE_BUFFER is valid
            // here, but examples all across the docs seem to do this
            VkDescriptorSetLayoutBinding storage_buffer_layout = {
                 (uint32_t)layout_bindings.size(),   // binding index
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // descriptor type
                 1,                                  // descriptor count
                 VK_SHADER_STAGE_COMPUTE_BIT,        // stage flags
                 nullptr                             // immutable samplers
            };
            layout_bindings.append(user_context, &storage_buffer_layout);
        }
        i++;
    }
    // Create the LayoutInfo struct
    VkDescriptorSetLayoutCreateInfo layout_info = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,  // structure type
        nullptr,                                              // pointer to a struct extending this info
        0,                                                    // flags
        (uint32_t)layout_bindings.size(),                     // binding count
        (VkDescriptorSetLayoutBinding*)layout_bindings.data() // pointer to layout bindings array
    };

    // Create the descriptor set layout
    VkResult result = vkCreateDescriptorSetLayout(device, &layout_info, 0, layout);
    if (result != VK_SUCCESS) {
        debug(user_context) << "vkCreateDescriptorSetLayout returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    return VK_SUCCESS;
}

WEAK VkResult vk_create_pipeline_layout(void* user_context, 
                                        VkDevice device,
                                        const VkAllocationCallbacks* alloc_callbacks,
                                        VkDescriptorSetLayout* descriptor_set_layout,
                                        VkPipelineLayout* pipeline_layout){

    debug(user_context)
        << "Vulkan: vk_create_pipeline_layout (user_context: " << user_context << ", "
        << "device: " << (void*)device << ", "
        << "descriptor_set_layout: " << (void*)descriptor_set_layout << ", "
        << "pipeline_layout: " << (void*)pipeline_layout << ")\n";


    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // structure type
        nullptr,                                        // pointer to a structure extending this
        0,                                              // flags
        1,                                              // number of descriptor sets
        descriptor_set_layout,                          // pointer to the descriptor sets
        0,                                              // number of push constant ranges
        nullptr                                         // pointer to push constant range structs
    };

    VkResult result = vkCreatePipelineLayout(device, &pipeline_layout_info, alloc_callbacks, pipeline_layout);
    if (result != VK_SUCCESS) {
        debug(user_context) << "vkCreatePipelineLayout returned " << vk_get_error_name(result) << "\n";
        return result;
    }
    return VK_SUCCESS;
}

WEAK int halide_vulkan_run(void *user_context,
                           void *state_ptr,
                           const char *entry_name,
                           int blocksX, int blocksY, int blocksZ,
                           int threadsX, int threadsY, int threadsZ,
                           int shared_mem_bytes,
                           size_t arg_sizes[],
                           void *args[],
                           int8_t arg_is_buffer[]) {
    debug(user_context)
        << "Vulkan: halide_vulkan_run (user_context: " << user_context << ", "
        << "entry: " << entry_name << ", "
        << "blocks: " << blocksX << "x" << blocksY << "x" << blocksZ << ", "
        << "threads: " << threadsX << "x" << threadsY << "x" << threadsZ << ", "
        << "shmem: " << shared_mem_bytes << "\n";

    VulkanContext ctx(user_context);
    if (ctx.error != VK_SUCCESS) {
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    // Running a Vulkan pipeline requires a large number of steps
    // and boilerplate:
    // 1. Create a descriptor set layout
    // 1a. Create the buffer for the scalar params
    // 2. Create a pipeline layout
    // 3. Create a compute pipeline
    // --- The above can be cached between invocations ---
    // 4. Create a descriptor set
    // 5. Set bindings for buffers in the descriptor set
    // 6. Create a command pool
    // 7. Create a command buffer from the command pool
    // 8. Begin the command buffer
    // 9. Bind the compute pipeline from #3
    // 10. Bind the descriptor set
    // 11. Add a dispatch to the command buffer
    // 12. End the command buffer
    // 13. Submit the command buffer to our command queue
    // --- The following isn't best practice, but it's in line
    //     with what we do in Metal etc. ---
    // 14. Wait until the queue is done with the command buffer

    uint32_t num_bindings = vk_count_bindings_for_descriptor_set(user_context, arg_sizes, args, arg_is_buffer);

    //// 1. Create a descriptor set layout
    VkDescriptorSetLayout descriptor_set_layout;
    VkResult result = vk_create_descriptor_set_layout(user_context, ctx.device, arg_sizes, args, arg_is_buffer, &descriptor_set_layout);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vk_create_descriptor_set_layout() failed! Unable to create shader module!\n";
        return result;
    }

    //// 1a. Create a buffer for the scalar parameters
    // First allocate memory, then map it and copy params, then create a buffer
    // and bind the allocation
    MemoryRegion* scalar_args_region = vk_create_scalar_uniform_buffer(user_context, ctx.allocator, arg_sizes, args, arg_is_buffer);
    if (scalar_args_region == nullptr) {
        error(user_context) << "Vulkan: vk_create_scalar_uniform_buffer() failed! Unable to create shader module!\n";
        return result;
    }

    VkBuffer* scalar_args_buffer = reinterpret_cast<VkBuffer*>(scalar_args_region->handle);
    if (scalar_args_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve scalar args buffer for device memory!\n";
        return halide_error_code_internal_error;
    }
    ///// 2. Create a pipeline layout
    VkPipelineLayout pipeline_layout;
    result = vk_create_pipeline_layout(user_context, ctx.device, ctx.allocator->callbacks(), &descriptor_set_layout, &pipeline_layout);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vk_create_pipeline_layout() failed! Unable to create shader module!\n";
        return result;
    }

    //// 3. Create a compute pipeline
    // Get the shader module

    VkShaderModule* shader_module = nullptr;
    bool found = compilation_cache.lookup(ctx.device, state_ptr, shader_module);
    halide_abort_if_false(user_context, found);
    halide_abort_if_false(user_context, shader_module != nullptr);
    VkComputePipelineCreateInfo compute_pipeline_info =
        {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,  // structure type
            nullptr,                                         // pointer to a structure extending this
            0,                                               // flags
            // VkPipelineShaderStageCreatInfo
            {
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // structure type
                nullptr,                                              //pointer to a structure extending this
                0,                                                    // flags
                VK_SHADER_STAGE_COMPUTE_BIT,                          // compute stage shader
                *shader_module,                                       // shader module
                entry_name,                                           // entry point name
                nullptr                                               // pointer to VkSpecializationInfo struct
            },
            pipeline_layout,  // pipeline layout
            0,                // base pipeline handle for derived pipeline
            0                 // base pipeline index for derived pipeline
        };

    VkPipeline pipeline;
    result = vkCreateComputePipelines(ctx.device, 0, 1, &compute_pipeline_info, 0, &pipeline);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkCreateComputePipelines returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 4. Create a descriptor set
    VkDescriptorPoolSize descriptor_pool_sizes[2] = {
        {
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // descriptor type
            1                                   // how many
        },
        {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // descriptor type
            num_bindings - 1                    // how many
        }};

    VkDescriptorPoolCreateInfo descriptor_pool_info =
        {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,  // struct type
            nullptr,                                        // point to struct extending this
            0,                                              // flags
            num_bindings,                                   // max numbewr of sets that can be allocated TODO:should this be 1?
            2,                                              // pool size count
            descriptor_pool_sizes                           // ptr to descriptr pool sizes
        };

    VkDescriptorPool descriptor_pool;
    result = vkCreateDescriptorPool(ctx.device, &descriptor_pool_info, 0, &descriptor_pool);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkCreateDescriptorPool returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    VkDescriptorSetAllocateInfo descriptor_set_info =
        {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,  // struct type
            nullptr,                                         // pointer to struct extending this
            descriptor_pool,                                 // pool from which to allocate sets
            1,                                               // number of descriptor sets
            &descriptor_set_layout                           // pointer to array of descriptor set layouts
        };

    VkDescriptorSet descriptor_set;
    result = vkAllocateDescriptorSets(ctx.device, &descriptor_set_info, &descriptor_set);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkAllocateDescriptorSets returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 5. Set bindings for buffers in the descriptor set
    const size_t HALIDE_MAX_VK_BINDINGS = 64;
    VkDescriptorBufferInfo descriptor_buffer_info[HALIDE_MAX_VK_BINDINGS];
    VkWriteDescriptorSet write_descriptor_set[HALIDE_MAX_VK_BINDINGS];

    // First binding will be the scalar params buffer
    descriptor_buffer_info[0] =
        {
            *scalar_args_buffer,  // the buffer
            0,                    // offset
            VK_WHOLE_SIZE         // range
        };
    write_descriptor_set[0] =
        {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  // struct type
            nullptr,                                 // pointer to struct extending this
            descriptor_set,                          // descriptor set to update
            0,                                       // binding
            0,                                       // array elem
            1,                                       // num to update
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,       // descriptor type
            nullptr,                                 // for images
            &(descriptor_buffer_info[0]),            // info for buffer
            nullptr                                  // for texel buffers
        };
    uint32_t num_bound = 1;
    for (size_t i = 0; arg_sizes[i] > 0; i++) {
        if (arg_is_buffer[i]) {
            halide_debug_assert(user_context, num_bound < HALIDE_MAX_VK_BINDINGS);

            // get the allocated region for the buffer
            MemoryRegion* device_region = reinterpret_cast<MemoryRegion*>(((halide_buffer_t *)args[i])->device);

            // retrieve the buffer from the region
            VkBuffer* device_buffer = reinterpret_cast<VkBuffer*>(device_region->handle);
            if (device_buffer == nullptr) {
                error(user_context) << "Vulkan: Failed to retrieve buffer for device memory!\n";
                return halide_error_code_internal_error;
            }

            descriptor_buffer_info[num_bound] =
                {
                    *device_buffer, // the buffer
                    0,              // offset
                    VK_WHOLE_SIZE   // range
                };
            write_descriptor_set[num_bound] =
                {
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  // struct type
                    nullptr,                                 // pointer to struct extending this
                    descriptor_set,                          // descriptor set to update
                    num_bound,                               // binding
                    0,                                       // array elem
                    1,                                       // num to update
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,       // descriptor type
                    nullptr,                                 // for images
                    &(descriptor_buffer_info[num_bound]),    // info for buffer
                    nullptr                                  // for texel buffers
                };
            num_bound++;
        }
    }

    halide_debug_assert(user_context, num_bound == num_bindings);
    vkUpdateDescriptorSets(ctx.device, num_bindings, write_descriptor_set, 0, nullptr);

    //// 6. Create a command pool
    // TODO: This should really be part of the acquire_context API
    // VkCommandPoolCreateInfo command_pool_info =
    //     {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,    // struct type
    //      nullptr,   // pointer to struct extending this
    //      0,     // flags.  may consider VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
    //      ctx.queue_family_index     // queue family index corresponding to the compute command queue
    //     };
    VkCommandPool command_pool;
    // result = vkCreateCommandPool(ctx.device, &command_pool_info, ctx.allocation_callbacks(), &command_pool);
    result = vk_create_command_pool(ctx.device, ctx.queue_family_index, ctx.allocation_callbacks(), &command_pool);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkCreateCommandPool returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 7. Create a command buffer from the command pool
    // VkCommandBufferAllocateInfo command_buffer_info =
    //     {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,    // struct type
    //      nullptr,    // pointer to struct extending this
    //      command_pool,  // command pool for allocation
    //      VK_COMMAND_BUFFER_LEVEL_PRIMARY,   // command buffer level
    //      1  // number to allocate
    //     };

    VkCommandBuffer command_buffer;
    //result = vkAllocateCommandBuffers(ctx.device, &command_buffer_info, &command_buffer);
    result = vk_create_command_buffer(ctx.device, command_pool, &command_buffer);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkAllocateCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 8. Begin the command buffer
    VkCommandBufferBeginInfo command_buffer_begin_info =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // struct type
            nullptr,                                      // pointer to struct extending this
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
            nullptr                                       // pointer to parent command buffer
        };

    result = vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkBeginCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 9. Bind the compute pipeline from #3
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    //// 10. Bind the descriptor set
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                            0, 1, &descriptor_set, 0, nullptr);

    //// 11. Add a dispatch to the command buffer
    // TODO: Is this right?
    vkCmdDispatch(command_buffer, blocksX, blocksY, blocksZ);

    //// 12. End the command buffer
    result = vkEndCommandBuffer(command_buffer);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkEndCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 13. Submit the command buffer to our command queue
    VkSubmitInfo submit_info =
        {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,  // struct type
            nullptr,                        // pointer to struct extending this
            0,                              // wait semaphore count
            nullptr,                        // semaphores
            nullptr,                        // pipeline stages where semaphore waits occur
            1,                              // how many command buffers to execute
            &command_buffer,                // the command buffers
            0,                              // number of semaphores to signal
            nullptr                         // the semaphores to signal
        };

    result = vkQueueSubmit(ctx.queue, 1, &submit_info, 0);

    if (result != VK_SUCCESS) {
        debug(user_context) << "vkQueueSubmit returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 14. Wait until the queue is done with the command buffer
    result = vkQueueWaitIdle(ctx.queue);
    if (result != VK_SUCCESS) {
        debug(user_context) << "vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    // Release the uniform buffer for the scalar args
    vk_destroy_scalar_uniform_buffer(user_context, ctx.allocator, scalar_args_region);

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif
    return 0;
}

WEAK int halide_vulkan_device_and_host_malloc(void *user_context, struct halide_buffer_t *buf) {
    return halide_default_device_and_host_malloc(user_context, buf, &vulkan_device_interface);
}

WEAK int halide_vulkan_device_and_host_free(void *user_context, struct halide_buffer_t *buf) {
    return halide_default_device_and_host_free(user_context, buf, &vulkan_device_interface);
}

WEAK int halide_vulkan_wrap_vk_buffer(void *user_context, struct halide_buffer_t *buf, uint64_t vk_buffer) {
    halide_debug_assert(user_context, buf->device == 0);
    if (buf->device != 0) {
        return -2;
    }
    buf->device = vk_buffer;
    buf->device_interface = &vulkan_device_interface;
    buf->device_interface->impl->use_module();

    return 0;
}

WEAK int halide_vulkan_detach_vk_buffer(void *user_context, halide_buffer_t *buf) {
    if (buf->device == 0) {
        return 0;
    }
    halide_debug_assert(user_context, buf->device_interface == &vulkan_device_interface);
    buf->device = 0;
    buf->device_interface->impl->release_module();
    buf->device_interface = nullptr;
    return 0;
}

WEAK uintptr_t halide_vulkan_get_vk_buffer(void *user_context, halide_buffer_t *buf) {
    if (buf->device == 0) {
        return 0;
    }
    halide_debug_assert(user_context, buf->device_interface == &vulkan_device_interface);
    return (uintptr_t)buf->device;
}

WEAK const struct halide_device_interface_t *halide_vulkan_device_interface() {
    return &vulkan_device_interface;
}

namespace {

__attribute__((destructor))
WEAK void
halide_vulkan_cleanup() {
    halide_vulkan_device_release(nullptr);
}

// --------------------------------------------------------------------------

}  // namespace

// --------------------------------------------------------------------------

}  // extern "C" linkage

// --------------------------------------------------------------------------

namespace Halide {
namespace Runtime {
namespace Internal {
namespace Vulkan {

// --------------------------------------------------------------------------

WEAK halide_device_interface_impl_t vulkan_device_interface_impl = {
    halide_use_jit_module,
    halide_release_jit_module,
    halide_vulkan_device_malloc,
    halide_vulkan_device_free,
    halide_vulkan_device_sync,
    halide_vulkan_device_release,
    halide_vulkan_copy_to_host,
    halide_vulkan_copy_to_device,
    halide_vulkan_device_and_host_malloc,
    halide_vulkan_device_and_host_free,
    halide_default_buffer_copy,
    halide_default_device_crop,
    halide_default_device_slice,
    halide_default_device_release_crop,
    halide_vulkan_wrap_vk_buffer,
    halide_vulkan_detach_vk_buffer,
};

WEAK halide_device_interface_t vulkan_device_interface = {
    halide_device_malloc,
    halide_device_free,
    halide_device_sync,
    halide_device_release,
    halide_copy_to_host,
    halide_copy_to_device,
    halide_device_and_host_malloc,
    halide_device_and_host_free,
    halide_buffer_copy,
    halide_device_crop,
    halide_device_slice,
    halide_device_release_crop,
    halide_device_wrap_native,
    halide_device_detach_native,
    nullptr,  // target capabilities.
    &vulkan_device_interface_impl};

// --------------------------------------------------------------------------

}  // namespace Vulkan
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

