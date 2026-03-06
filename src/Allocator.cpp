#include "header/Allocator.hpp"

#include <utility>
using namespace zeza;

class AllocatorException : public std::exception {
    std::string msg;
public:
    explicit AllocatorException(std::string message) : msg(std::move(message)) {}
    virtual const char *what() const noexcept override { return msg.c_str(); }
};

/////////////////////////////////////////////////////////////////////
/////////////////**************************************//////////////
/////////////////**  CPU-Allocator Struct Utilities  **//////////////
/////////////////**************************************//////////////
/////////////////////////////////////////////////////////////////////
;
void* CPUAllocator::allocate(uint64_t bytes){
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;

    #if defined(_MSC_VER) || defined(_WIN32)
    ptr = _aligned_malloc(bytes, ALIGNMENT);
    #elif defined(__cplusplus) && __cplusplus >= 201703L
    ptr = std::aligned_alloc(ALIGNMENT, bytes);
    #else
    if (posix_memalign(&ptr, ALIGNMENT, bytes) != 0) ptr = nullptr;
    #endif
    if (!ptr){
        // TODO : Make it try, and catch if can't allocate, to
        //  try again with a different size with delays.
        throw AllocatorException("[CPU Allocator] Failed To Fetch _aligned_malloc().");
    }

    // TODO: should I do zero initialize? not for now; for speed.
    // TODO: when I feel like, it's memset.

    std::string msg = std::format("[CPU Allocator] Allocated Memory: fulfilled the requested {} B. ", std::to_string(bytes));
    LOG_DEBUG(msg);
    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    #if defined(_MSC_VER) || defined(_WIN32)
    _aligned_free(ptr);
    #else
    free(ptr);
    #endif

    std::string msg = std::format("[CPU Allocator] Deallocated Memory.");
    LOG_DEBUG(msg);
}

Device CPUAllocator::device_type() const { return Device::CPU; }


//////////////////////////////////////////////////////////////////////////
/////////////////******************************************///////////////
/////////////////**  Allocator-Manager Struct Utilities  **///////////////
/////////////////******************************************///////////////
//////////////////////////////////////////////////////////////////////////

AllocatorManager::AllocatorManager() {
    register_allocator(Device::CPU, &cpu_alloc);
    //Todo: to be added when moved to CUDA.
    register_allocator(Device::CUDA, nullptr);
}

AllocatorManager& AllocatorManager::get() {
    static AllocatorManager instance;
    return instance;
}

Allocator* AllocatorManager::get_allocator(Device dev) {
    Allocator* alloc = allocators[static_cast<int>(dev)];
    if (!alloc) {
        throw std::runtime_error("Allocator not initialized for this device (Did you forget to link CUDA?).");
    }
    return alloc;
}

void AllocatorManager::register_allocator(Device dev, Allocator* alloc) {
    allocators[static_cast<int>(dev)] = alloc;
}
