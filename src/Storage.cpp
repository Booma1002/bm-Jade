#include "header/Storage.hpp"
using namespace zeza;

///////////////////////////////////////////////////////////////
/////////////////*******************************///////////////
/////////////////**  Storage Class Utilities  **///////////////
/////////////////*******************************///////////////
///////////////////////////////////////////////////////////////

uint64_t constexpr Storage::align(uint64_t base, uint64_t n){
    return (n+(base-1))& ~(base-1);
}

Storage::Storage(uint64_t n_elements, uint64_t i_size, Device dev, uint64_t alignment)
: num_elements(n_elements), item_size(i_size), device(dev), alignment(alignment) {
    allocator = AllocatorManager::get().get_allocator(dev);
    uint64_t request = static_cast<uint64_t>(n_elements * 1.25);
    capacity_bytes = request * item_size;
    capacity_bytes = align(alignment, capacity_bytes);
    memory = allocator->allocate(capacity_bytes);

    if (dev == Device::CPU) {
        std::memset(memory, 0, capacity_bytes);
    }
    else{
        // TODO: GPU needs cudaMemset
        throw std::runtime_error("CUDAMemset not yet implemented");
    }
    std::string msg = std::format("[Storage] Storage Alloc: {} B on Hardware[{}]", std::to_string(capacity_bytes), std::to_string((int)dev));
    LOG_INFO(msg);
}

Storage::~Storage() {
    free_memory();
}

void Storage::ensure_capacity(uint64_t needed_elements, double extra_scale) {
    uint64_t needed_bytes = needed_elements * item_size;
    if (needed_bytes <= capacity_bytes){
        std::string msg = std::format("[Storage] Ensured The Requested {}  Elements; Size Already Satisfied.", std::to_string(needed_elements));
        LOG_INFO(msg);
        return;
    }

    auto new_cap_N = static_cast<uint64_t>(static_cast<double>(needed_elements) * extra_scale);
    uint64_t new_bytes = new_cap_N * item_size;

    new_bytes = align(alignment, new_bytes);
    void* new_memory = allocator->allocate(new_bytes);

    if (device == Device::CPU) {
        std::memcpy(new_memory, memory, capacity_bytes);
        std::memset(static_cast<uint8_t*>(new_memory) + capacity_bytes, 0, new_bytes - capacity_bytes);
    } else {
        // TODO: Add cudaMemcpy logic here later
        throw std::runtime_error("GPU resize not yet implemented");
    }

    allocator->deallocate(memory);
    memory = new_memory;
    capacity_bytes = new_bytes;
    std::string msg = std::format("[Storage] Storage Grow: {}B", std::to_string(new_bytes));
    LOG_DEBUG(msg);
    msg = std::format("[Storage] Ensured The Requested {} Elements.", std::to_string(needed_elements));
    LOG_INFO(msg);
}

void* Storage::get_offset_ptr(uint64_t element_offset) const {
    return static_cast<uint8_t*>(memory) + (element_offset * item_size);
};

void Storage::free_memory() {
    if (memory && allocator) {
        allocator->deallocate(memory);
        memory = nullptr;
        std::string msg = "[Storage] Storage Freed. 0B";
        LOG_INFO(msg);
    }
}


///////////////////////////////////////////////////////////////////
/////////////////***********************************///////////////
/////////////////**  Storage Class Encapsulation  **///////////////
/////////////////***********************************///////////////
///////////////////////////////////////////////////////////////////
uint64_t Storage::size() const { return num_elements; }

uint64_t Storage::capacity() const { return capacity_bytes; }

uint64_t Storage::get_item_size() const { return item_size; }

Device Storage::get_device() const { return device; }


