#ifndef HALIDE_RUNTIME_VULKAN_EXTENSIONS_H
#define HALIDE_RUNTIME_VULKAN_EXTENSIONS_H

#include "vulkan_internal.h"

// --

namespace Halide {
namespace Runtime {
namespace Internal {
namespace Vulkan {

// --

WEAK char layer_names[1024];
WEAK ScopedSpinLock::AtomicFlag layer_names_lock = 0;
WEAK bool layer_names_initialized = false;

WEAK char extension_names[1024];
WEAK ScopedSpinLock::AtomicFlag extension_names_lock = 0;
WEAK bool extension_names_initialized = false;

WEAK char device_type[256];
WEAK ScopedSpinLock::AtomicFlag device_type_lock = 0;
WEAK bool device_type_initialized = false;

WEAK char build_options[1024];
WEAK ScopedSpinLock::AtomicFlag build_options_lock = 0;
WEAK bool build_options_initialized = false;

// --

WEAK void vk_set_layer_names_internal(const char *n) {
    if (n) {
        size_t buffer_size = sizeof(layer_names) / sizeof(layer_names[0]);
        strncpy(layer_names, n, buffer_size);
        layer_names[buffer_size - 1] = '\0';
    } else {
        layer_names[0] = '\0';
    }
    layer_names_initialized = true;
}

WEAK const char *vk_get_layer_names_internal(void *user_context) {
    if (!layer_names_initialized) {
        const char *value = getenv("HL_VK_LAYERS");
        if (value == nullptr) { value = getenv("VK_INSTANCE_LAYERS"); }
        vk_set_layer_names_internal(value);
    }
    return layer_names;
}

WEAK void vk_set_extension_names_internal(const char *n) {
    if (n) {
        size_t buffer_size = sizeof(extension_names) / sizeof(extension_names[0]);
        strncpy(extension_names, n, buffer_size);
        extension_names[buffer_size - 1] = 0;
    } else {
        extension_names[0] = 0;
    }
    extension_names_initialized = true;
}

WEAK const char *vk_get_extension_names_internal(void *user_context) {
    if (!extension_names_initialized) {
        const char *name = getenv("HL_VK_EXTENSIONS");
        vk_set_extension_names_internal(name);
    }
    return extension_names;
}

WEAK void vk_set_device_type_internal(const char *n) {
    if (n) {
        size_t buffer_size = sizeof(device_type) / sizeof(device_type[0]);
        strncpy(device_type, n, buffer_size);
        device_type[buffer_size - 1] = 0;
    } else {
        device_type[0] = 0;
    }
    device_type_initialized = true;
}

WEAK const char *vk_get_device_type_internal(void *user_context) {
    if (!device_type_initialized) {
        const char *name = getenv("HL_VK_DEVICE_TYPE");
        vk_set_device_type_internal(name);
    }
    return device_type;
}

WEAK void vk_set_build_options_internal(const char *n) {
    if (n) {
        size_t buffer_size = sizeof(build_options) / sizeof(build_options[0]);
        strncpy(build_options, n, buffer_size);
        build_options[buffer_size - 1] = 0;
    } else {
        build_options[0] = 0;
    }
    build_options_initialized = true;
}

WEAK const char *vk_get_build_options_internal(void *user_context) {
    if (!build_options_initialized) {
        const char *name = getenv("HL_VK_BUILD_OPTIONS");
        vk_set_build_options_internal(name);
    }
    return build_options;
}

WEAK uint32_t vk_get_requested_layers(void *user_context, StringTable &layer_table) {
    ScopedSpinLock lock(&layer_names_lock);
    const char *layer_names = vk_get_layer_names_internal(user_context);
    return layer_table.parse(user_context, layer_names, HL_VK_ENV_DELIM);
}

WEAK uint32_t vk_get_required_instance_extensions(void *user_context, StringTable &ext_table) {
    const uint32_t required_ext_count = 1;
    const char *required_ext_table[] = {"VK_KHR_get_physical_device_properties2"};

    ext_table.reserve(user_context, required_ext_count);
    for (uint32_t n = 0; n < required_ext_count; ++n) {
        ext_table.assign(user_context, n, required_ext_table[n]);
    }
    return required_ext_count;
}

WEAK uint32_t vk_get_supported_instance_extensions(void *user_context, StringTable &ext_table) {

    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceExtensionProperties");

    if (vkEnumerateInstanceExtensionProperties == nullptr) {
        debug(user_context) << "Vulkan: Missing vkEnumerateInstanceExtensionProperties proc address! Invalid loader?!\n";
        return 0;
    }

    uint32_t avail_ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &avail_ext_count, nullptr);
    debug(user_context) << "Vulkan: vkEnumerateInstanceExtensionProperties found  " << avail_ext_count << " extensions ...\n";

    if (avail_ext_count) {
        BlockStorage<VkExtensionProperties> extension_properties(user_context, avail_ext_count);
        extension_properties.resize(user_context, avail_ext_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &avail_ext_count, extension_properties.data());
        for (uint32_t n = 0; n < avail_ext_count; ++n) {
            debug(user_context) << "    extension: " << extension_properties[n].extensionName << "\n";
        }

        ext_table.reserve(user_context, avail_ext_count);
        for (uint32_t n = 0; n < avail_ext_count; ++n) {
            ext_table.assign(user_context, n, extension_properties[n].extensionName);
        }
    }

    return avail_ext_count;
}

WEAK uint32_t vk_get_required_device_extensions(void *user_context, StringTable &ext_table) {
    const uint32_t required_ext_count = 0;
    const char *required_ext_table[] = {0};

    ext_table.reserve(user_context, required_ext_count);
    for (uint32_t n = 0; n < required_ext_count; ++n) {
        ext_table.assign(user_context, n, required_ext_table[n]);
    }
    return required_ext_count;
}

WEAK uint32_t vk_get_supported_device_extensions(void *user_context, VkPhysicalDevice physical_device, StringTable &ext_table) {

    if (vkEnumerateDeviceExtensionProperties == nullptr) {
        debug(user_context) << "Vulkan: Missing vkEnumerateDeviceExtensionProperties proc address! Invalid loader?!\n";
        return 0;
    }

    uint32_t avail_ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &avail_ext_count, nullptr);
    debug(user_context) << "Vulkan: vkEnumerateDeviceExtensionProperties found  " << avail_ext_count << " extensions ...\n";

    if (avail_ext_count) {
        BlockStorage<VkExtensionProperties> extension_properties(user_context, avail_ext_count);
        extension_properties.resize(user_context, avail_ext_count);
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &avail_ext_count, extension_properties.data());
        for (uint32_t n = 0; n < avail_ext_count; ++n) {
            debug(user_context) << "    extension: " << extension_properties[n].extensionName << "\n";
        }

        ext_table.reserve(user_context, avail_ext_count);
        for (uint32_t n = 0; n < avail_ext_count; ++n) {
            ext_table.assign(user_context, n, extension_properties[n].extensionName);
        }
    }

    return avail_ext_count;
}

WEAK bool vk_validate_required_extension_support(void *user_context,
                                                 const StringTable &required_extensions,
                                                 const StringTable &supported_extensions) {
    bool validated = true;
    for (uint32_t n = 0; n < required_extensions.size(); ++n) {
        const char *extension = required_extensions[n];
        if (!supported_extensions.contains(extension)) {
            debug(user_context) << "Vulkan: Missing required extension: '" << extension << "'! \n";
            validated = false;
        }
    }
    return validated;
}

}  // namespace Vulkan
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

using namespace Halide::Runtime::Internal::Vulkan;

extern "C" {

WEAK void halide_vulkan_set_layer_names(const char *n) {
    ScopedSpinLock lock(&layer_names_lock);
    vk_set_layer_names_internal(n);
}

WEAK const char *halide_vulkan_get_layer_names(void *user_context) {
    ScopedSpinLock lock(&layer_names_lock);
    return vk_get_layer_names_internal(user_context);
}

WEAK void halide_vulkan_set_extension_names(const char *n) {
    ScopedSpinLock lock(&extension_names_lock);
    vk_set_extension_names_internal(n);
}

WEAK const char *halide_vulkan_get_extension_names(void *user_context) {
    ScopedSpinLock lock(&extension_names_lock);
    return vk_get_extension_names_internal(user_context);
}

WEAK void halide_vulkan_set_device_type(const char *n) {
    ScopedSpinLock lock(&device_type_lock);
    vk_set_device_type_internal(n);
}

WEAK const char *halide_vulkan_get_device_type(void *user_context) {
    ScopedSpinLock lock(&device_type_lock);
    return vk_get_device_type_internal(user_context);
}

WEAK void halide_vulkan_set_build_options(const char *n) {
    ScopedSpinLock lock(&build_options_lock);
    vk_set_build_options_internal(n);
}

WEAK const char *halide_vulkan_get_build_options(void *user_context) {
    ScopedSpinLock lock(&build_options_lock);
    return vk_get_build_options_internal(user_context);
}

}  // extern "C"

#endif  // HALIDE_RUNTIME_VULKAN_EXTENSIONS_H