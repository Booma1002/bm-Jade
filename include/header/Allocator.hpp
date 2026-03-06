#define _POSIX_C_SOURCE 200809L
#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <new>
#include <iostream>
#include "Enums.hpp"
#if defined(_MSC_VER) || defined(_WIN32)
#include <malloc.h>
#endif
namespace zeza {
/**
 * @brief Exception thrown during failed low-level memory allocations.
 * @usage Triggered when OS-specific aligned allocation routines (like `_aligned_malloc` or
 * `posix_memalign`) fail to secure the requested byte block.
 */
    class AllocatorException : public std::exception {
        std::string msg;
    public:
        explicit AllocatorException(std::string message) : msg(std::move(message)) {}
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };


///////////////////////////////////////////////////////////////////////
/////////////////***************************************///////////////
/////////////////**  Allocator Struct Initialization  **///////////////
/////////////////***************************************///////////////
///////////////////////////////////////////////////////////////////////
    ;

/**
 * @brief Abstract base class for hardware-specific memory allocators.
 * Decouples physical memory acquisition from the logical `Storage` containers.
 * This enables seamless swapping between CPU RAM, Pinned Host Memory, and GPU VRAM
 * without modifying the higher-level Tile architecture.
 */
    struct Allocator {

        /**
     * @brief Acquires a raw, aligned block of physical memory.
     * @warning The returned memory block is strictly uninitialized. The caller
     * (typically `Storage`) is fully responsible for zeroing or filling the memory to prevent
     * garbage data propagation.
     * @param bytes The exact footprint size in bytes to allocate.
     * @return A raw void pointer to the start of the memory block, or nullptr if bytes == 0.
     */
        virtual void *allocate(size_t bytes) = 0;

        /**
     * @brief Returns a previously allocated block of memory to the operating system.
     * @warning UB Warning: Passing a null pointer, a pointer modified by arithmetic, or
     * a pointer allocated by a different device's allocator will cause immediate segfaults
     * or heap corruption.
     * @param ptr The exact void pointer originally returned by `allocate()`.
     */
        virtual void deallocate(void *ptr) = 0;


        /**
     * @brief Identifies the hardware domain this allocator controls.
     * @return The `Device` enum associated with this allocator's memory pool.
     */
        virtual Device device_type() const = 0;

        virtual ~Allocator() = default;
    };

///////////////////////////////////////////////////////////////////////////
/////////////////*******************************************///////////////
/////////////////**  CPU-Allocator Struct Initialization  **///////////////
/////////////////*******************************************///////////////
///////////////////////////////////////////////////////////////////////////
    ;

/**
 * @brief Concrete allocator for standard Host (CPU) memory.
 * Utilizes OS-specific aligned allocation routines to guarantee strict memory boundaries.
 * This is critical for cache-line optimization and enabling AVX2/AVX-512 vectorization
 * instructions during continuous strided access.
 */
    struct CPUAllocator : public Allocator {
/////////////////////////////////////////////////////////////////////
/////////////////**************************************//////////////
/////////////////**  CPU-Allocator Struct Utilities  **//////////////
/////////////////**************************************//////////////
/////////////////////////////////////////////////////////////////////
        ;
/**
 * @brief The strict byte-boundary alignment constraint.
 * Currently locked to 32 bytes to ensure safe loading into 256-bit SIMD registers.
 */
        static constexpr size_t ALIGNMENT = 64;

        void *allocate(uint64_t bytes) override;

        void deallocate(void *ptr) override;

        [[nodiscard]] Device device_type() const override;
    };


///////////////////////////////////////////////////////////////////////////////
/////////////////***********************************************///////////////
/////////////////**  Allocator-Manager Struct Initialization  **///////////////
/////////////////***********************************************///////////////
///////////////////////////////////////////////////////////////////////////////
    ;

/**
 * @brief Global Singleton registry for routing allocations to hardware targets.
 * Maintains a centralized dispatch table mapped to the `Device` enum.
 * @interface When a `Storage`
 * object requests memory, it queries this manager to abstract away the underlying
 * hardware-specific allocation APIs (like CPU vs CUDA).
 */
    class AllocatorManager {

//////////////////////////////////////////////////////////////////////////
/////////////////******************************************///////////////
/////////////////**  Allocator-Manager Struct Utilities  **///////////////
/////////////////******************************************///////////////
//////////////////////////////////////////////////////////////////////////
    private:
        Allocator *allocators[static_cast<int>(Device::MAX_DEVICES)];
        CPUAllocator cpu_alloc;
        //GPUAllocator gpu_alloc;

        AllocatorManager();

    public:
        /**
     * @brief Retrieves the global static instance of the manager.
     * @note Guaranteed thread-safe initialization under C++11 standard rules.
     * @return Reference to the singleton instance.
     */
        static AllocatorManager &get();

        /**
     * @brief Fetches the active hardware-specific memory allocator.
     * @throws std::runtime_error If the requested device backend (e.g., CUDA) was never
     * registered or linked during framework initialization.
     * @param dev The hardware target to allocate on.
     * @return A pointer to the concrete allocator instance.
     */
        Allocator *get_allocator(Device dev);

        /**
     * @brief Injects a custom or hardware-specific allocator into the global dispatch table.
     * Used during framework boot-up to dynamically attach GPU/TPU allocators without
     * tightly coupling the core C++ headers to proprietary SDKs like CUDA.
     * @param dev The hardware domain the allocator manages.
     * @param alloc Pointer to the concrete allocator instance.
     */
        void register_allocator(Device dev, Allocator *alloc);
    };


}// namespace zeza