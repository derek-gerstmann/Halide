#ifndef HALIDE_RUNTIME_REGION_ALLOCATOR_H
#define HALIDE_RUNTIME_REGION_ALLOCATOR_H

#include "linked_list.h"
#include "memory_resources.h"

namespace Halide {
namespace Runtime {
namespace Internal {

// --

/** Allocator class interface for sub-allocating a contiguous
 * memory block into smaller regions of memory. This class only 
 * manages the address creation for the regions -- allocation 
 * callback functions are used to request the memory from the 
 * necessary system or API calls. This class is intended to be 
 * used inside of a higher level memory management class that 
 * provides thread safety, policy management and API 
 * integration for a specific runtime API (eg Vulkan, OpenCL, etc) 
*/
class RegionAllocator {

    // disable copy constructors and assignment
    RegionAllocator(const RegionAllocator &) = delete;
    RegionAllocator &operator=(const RegionAllocator &) = delete;

public:
    static const uint32_t InvalidEntry = uint32_t(-1);
    typedef MemoryArena<BlockRegion> BlockRegionArena;

    struct MemoryAllocators {
        SystemMemoryAllocator* system = nullptr;
        MemoryRegionAllocator* region = nullptr;
    };

    RegionAllocator();
    RegionAllocator(void* user_context, BlockResource* block, const MemoryAllocators& ma);
    ~RegionAllocator();

    // Initializes a new instance     
    void initialize(void* user_context, BlockResource* block, const MemoryAllocators& ma);

    // Factory methods for creation / destruction
    static RegionAllocator* create(void* user_context, BlockResource* block, const MemoryAllocators& ma);
    static void destroy(void* user_context, RegionAllocator* region_allocator);

    // Returns the allocator class instance for the given allocation (or nullptr)
    static RegionAllocator* find_allocator(void* user_context, MemoryRegion* memory_region);

    // Public interface methods
    MemoryRegion* reserve(void *user_context, size_t size, size_t alignment);
    void reclaim(void *user_context, MemoryRegion* memory_region);
    bool collect(void* user_context); //< returns true if any blocks were removed
    void destroy(void *user_context);

    // Returns the currently managed block resource
    BlockResource* block_resource() const;

private:

    // Search through allocated block regions (Best-Fit)    
    BlockRegion* find_block_region(void* user_context, size_t size, size_t alignment);

    // Returns true if neighbouring block regions to the given region can be coalesced into one
    bool can_coalesce(BlockRegion* region);

    // Merges available neighbouring block regions into the given region
    BlockRegion* coalesce_block_regions(void* user_context, BlockRegion* region);

    // Returns true if the given region can be split to accomadate the given size
    bool can_split(BlockRegion* region, size_t size);

    // Splits the given block region into a smaller region to accomadate the given size, followed by empty space for the remaining 
    BlockRegion* split_block_region(void* user_context, BlockRegion* region, size_t size, size_t alignment);

    // Creates a new block region and adds it to the region list 
    BlockRegion* create_block_region(void* user_context, size_t offset, size_t size);

    // Creates a new block region and adds it to the region list 
    void destroy_block_region(void* user_context, BlockRegion* region);

    // Invokes the allocation callback to allocate memory for the block region
    void alloc_block_region(void* user_context, BlockRegion* region);

    // Invokes the deallocation callback to free memory for the block region
    void free_block_region(void* user_context, BlockRegion* region);

private:
    BlockResource* block;
    BlockRegionArena arena;
    MemoryAllocators allocators;
};

RegionAllocator* RegionAllocator::create(void* user_context, BlockResource* block_resource, const MemoryAllocators& allocators) {
    halide_abort_if_false(user_context, allocators.system != nullptr);
    RegionAllocator* result = reinterpret_cast<RegionAllocator*>(
        allocators.system->allocate(user_context, sizeof(RegionAllocator))
    );

    if(result == nullptr) {
        error(user_context) << "RegionAllocator: Failed to create instance! Out of memory!\n";
        return nullptr; 
    }

    result->initialize(user_context, block_resource, allocators);
    return result;
}

void RegionAllocator::destroy(void* user_context, RegionAllocator* instance) {
    halide_abort_if_false(user_context, instance != nullptr);
    const MemoryAllocators& allocators = instance->allocators;
    instance->destroy(user_context);
    halide_abort_if_false(user_context, allocators.system != nullptr);
    allocators.system->deallocate(user_context, instance);
}

RegionAllocator::RegionAllocator() :
    block(nullptr),
    arena(nullptr),
    allocators() {
}

RegionAllocator::~RegionAllocator() {
    destroy(nullptr);
}

void RegionAllocator::initialize(void* user_context, BlockResource* mb, const MemoryAllocators& ma) {
    block = mb;
    allocators = ma;
    arena.initialize(user_context, { BlockRegionArena::default_capacity, 0 }, allocators.system);
    block->allocator = this;
    block->regions = create_block_region(user_context, 0, block->size);
}

MemoryRegion* RegionAllocator::reserve(void *user_context, size_t size, size_t alignment) {
    halide_abort_if_false(user_context, size > 0);
    size_t remaining = block->size - block->reserved;
    if(remaining < size) { 
        debug(user_context) << "RegionAllocator: Unable to reserve more memory from block " 
                            << "-- requested size (" << (int32_t)(size) << " bytes) "
                            << "greater than available (" << (int32_t)(remaining) << " bytes)!\n";    
        return nullptr; 
    }

    BlockRegion* block_region = find_block_region(user_context, size, alignment);
    if(block_region == nullptr) {
        debug(user_context) << "RegionAllocator: Failed to locate region for requested size (" 
                            << (int32_t)(size) << " bytes)!\n";

        return nullptr; 
    }
    
    if(can_split(block_region, size)) {
        debug(user_context) << "RegionAllocator: Splitting region of size ( " << (int32_t)(block_region->size) << ") "
                            << "to accomodate requested size (" << (int32_t)(size) << " bytes)!\n";

        split_block_region(user_context, block_region, size, alignment);
    }

    alloc_block_region(user_context, block_region);    
    return reinterpret_cast<MemoryRegion*>(block_region);
}

void RegionAllocator::reclaim(void* user_context, MemoryRegion* memory_region) {
    BlockRegion* block_region = reinterpret_cast<BlockRegion*>(memory_region);
    halide_abort_if_false(user_context, block_region != nullptr);
    halide_abort_if_false(user_context, block_region->block_ptr == block);
    free_block_region(user_context, block_region);
    if(can_coalesce(block_region)) {
        block_region = coalesce_block_regions(user_context, block_region);
    }
}

RegionAllocator* RegionAllocator::find_allocator(void* user_context, MemoryRegion* memory_region) {
    BlockRegion* block_region = reinterpret_cast<BlockRegion*>(memory_region);
    halide_abort_if_false(user_context, block_region != nullptr);
    halide_abort_if_false(user_context, block_region->block_ptr != nullptr);
    return block_region->block_ptr->allocator;
}

BlockRegion* RegionAllocator::find_block_region(void* user_context, size_t size, size_t alignment) {
    BlockRegion* result = nullptr;
    for(BlockRegion* block_region = block->regions; block_region != nullptr; block_region = block_region->next_ptr) {

        if(block_region->status != AllocationStatus::Available) {
            continue;
        }

        if(size > block_region->size) {
            continue;
        }

        size_t actual_size = aligned_size(block_region->offset, size, alignment);
        if(actual_size > block_region->size) {
            continue;
        }

        if((actual_size + block->reserved) < block->size) {
            result = block_region; // best-fit!
            break;
        }
    }
    return result;
}

bool RegionAllocator::can_coalesce(BlockRegion* block_region) {
    if(block_region == nullptr) { return false; }
    if(block_region->prev_ptr && (block_region->prev_ptr->status == AllocationStatus::Available)) {
        return true;
    }
    if(block_region->next_ptr && (block_region->next_ptr->status == AllocationStatus::Available)) {
        return true;
    }
    return false;
}

BlockRegion* RegionAllocator::coalesce_block_regions(void* user_context, BlockRegion* block_region) {

    if(block_region->prev_ptr && (block_region->prev_ptr->status == AllocationStatus::Available)) {
        BlockRegion* prev_region = block_region->prev_ptr;

        debug(user_context) << "RegionAllocator: Coalescing "
                            << "previous region (offset=" << prev_region->offset << " size=" << (int32_t)(prev_region->size) << " bytes) " 
                            << "into current region (offset=" << block_region->offset << " size=" << (int32_t)(block_region->size) << " bytes)\n!";

        prev_region->next_ptr = block_region->next_ptr;
        if(block_region->next_ptr) {
            block_region->next_ptr->prev_ptr = prev_region;
        }
        prev_region->size += block_region->size;
        destroy_block_region(user_context, block_region);
        block_region = prev_region;
    }

    if(block_region->next_ptr && (block_region->next_ptr->status == AllocationStatus::Available)) {
        BlockRegion* next_region = block_region->next_ptr;

        debug(user_context) << "RegionAllocator: Coalescing "
                            << "next region (offset=" << next_region->offset << " size=" << (int32_t)(next_region->size) << " bytes) " 
                            << "into current region (offset=" << block_region->offset << " size=" << (int32_t)(block_region->size) << " bytes)!\n";

        if(next_region->next_ptr) {
            next_region->next_ptr->prev_ptr = block_region;
        }
        block_region->next_ptr = next_region->next_ptr;
        block_region->size += next_region->size;
        destroy_block_region(user_context, next_region);
    }

    return block_region;
}

bool RegionAllocator::can_split(BlockRegion* block_region, size_t size) {
    return (block_region && (block_region->size > size));
}

BlockRegion* RegionAllocator::split_block_region(void* user_context, BlockRegion* block_region, size_t size, size_t alignment){

    size_t adjusted_size = aligned_size(block_region->offset, size, alignment);
    size_t adjusted_offset = aligned_offset(block_region->offset, alignment);

    size_t empty_offset = adjusted_offset + size;
    size_t empty_size = block_region->size - adjusted_size;

    debug(user_context) << "RegionAllocator: Splitting "
                        << "current region (offset=" << block_region->offset << " size=" << (int32_t)(block_region->size) << " bytes) " 
                        << "to create empty region (offset=" << empty_offset << " size=" << (int32_t)(empty_size) << " bytes)!\n";


    BlockRegion* next_region = block_region->next_ptr;
    BlockRegion* empty_region = create_block_region(user_context, empty_offset, empty_size);
    empty_region->next_ptr = next_region;
    if(next_region) { 
        next_region->prev_ptr = empty_region;
    }
    block_region->next_ptr = empty_region;
    block_region->size = size;
    return empty_region;
}

BlockRegion* RegionAllocator::create_block_region(void* user_context, size_t offset, size_t size) {
    BlockRegion* block_region = arena.reserve(user_context);
    memset(block_region, 0, sizeof(BlockRegion));
    block_region->offset = offset;
    block_region->size = size;
    block_region->status = AllocationStatus::Available;
    block_region->block_ptr = block;
    return block_region;
}

void RegionAllocator::destroy_block_region(void* user_context, BlockRegion* block_region) {
    free_block_region(user_context, block_region);
    arena.reclaim(user_context, block_region);
}

void RegionAllocator::alloc_block_region(void* user_context, BlockRegion* block_region) {
    debug(user_context) << "RegionAllocator: Allocating region of size ( " << (int32_t)(block_region->size) << ") bytes)!\n";
    halide_abort_if_false(user_context, allocators.region != nullptr);
    halide_abort_if_false(user_context, block_region->status == AllocationStatus::Available);
    MemoryRegion* memory_region = reinterpret_cast<MemoryRegion*>(block_region);
    allocators.region->allocate(user_context, memory_region);
    block_region->status = AllocationStatus::InUse;
    block->reserved += block_region->size;
}

void RegionAllocator::free_block_region(void* user_context, BlockRegion* block_region) {
    if(block_region->status == AllocationStatus::InUse) {
        debug(user_context) << "RegionAllocator: Freeing region of size ( " << (int32_t)(block_region->size) << ") bytes)!\n";
        halide_abort_if_false(user_context, allocators.region != nullptr);
        MemoryRegion* memory_region = reinterpret_cast<MemoryRegion*>(block_region);
        allocators.region->deallocate(user_context, memory_region);
        block->reserved -= block_region->size;
    }
    block_region->status = AllocationStatus::Available;
}

bool RegionAllocator::collect(void* user_context) {
    bool result = false;
    for(BlockRegion* block_region = block->regions; block_region != nullptr; block_region = block_region->next_ptr) {
        if(block_region->status == AllocationStatus::Available) {   
            if(can_coalesce(block_region)) {
                block_region = coalesce_block_regions(user_context, block_region);
                result = true;
            }
        }
    }
    return result;
}

void RegionAllocator::destroy(void* user_context) {
    for(BlockRegion* block_region = block->regions; block_region != nullptr; ) {
        
        if(block_region->next_ptr == nullptr) {
            destroy_block_region(user_context, block_region);
            block_region = nullptr;
        } else {
            BlockRegion* prev_region = block_region;
            block_region = block_region->next_ptr;
            destroy_block_region(user_context, prev_region);
        }
    }
    block->regions = nullptr;
    block->reserved = 0;
    arena.destroy(user_context);
}

BlockResource* RegionAllocator::block_resource() const {
    return block;
}

// --
    
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

#endif  // HALIDE_RUNTIME_REGION_ALLOCATOR_H