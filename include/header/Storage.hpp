#pragma once
#include <memory>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include "Allocator.hpp"

namespace zeza {
////////////////////////////////////////////////////////////////////
/////////////////************************************///////////////
/////////////////**  Storage Class Initialization  **///////////////
/////////////////************************************///////////////
////////////////////////////////////////////////////////////////////
    ;

/**
 *
 */
    class Storage {
//////////////////////////////////////////////////////////////
/////////////////******************************///////////////
/////////////////**  Storage Class Settings  **///////////////
/////////////////******************************///////////////
//////////////////////////////////////////////////////////////
    private:
        void *memory = nullptr;
        Allocator *allocator = nullptr;
        uint64_t capacity_bytes = 0;
        uint64_t num_elements = 0;
        uint64_t item_size = 0;
        Device device = Device::CPU;
        uint64_t alignment = 64;

        static constexpr uint64_t align(uint64_t base, uint64_t n);

///////////////////////////////////////////////////////////////
/////////////////*******************************///////////////
/////////////////**  Storage Class Utilities  **///////////////
/////////////////*******************************///////////////
///////////////////////////////////////////////////////////////
    public:

        /**
         *
         * @param n_elements
         * @param i_size
         * @param dev
         * @param alignment
         */
        Storage(uint64_t n_elements, uint64_t i_size, Device dev = Device::CPU, uint64_t alignment = 32);

        /**
         *
         */
        ~Storage();

        /**
         *
         * @param needed_elements
         * @param extra_scale
         * @param fill
         * @param new_value
         */
        void ensure_capacity(uint64_t needed_elements, double extra_scale = 1.5);

        void *get_offset_ptr(uint64_t element_offset) const;

////////////////////////////////////////////////////////
/////////////////************************///////////////
/////////////////**  Storage Indexing  **///////////////
/////////////////************************///////////////
////////////////////////////////////////////////////////
        ;

        /**
         * Get Reference To Physical Storage In The Heap.
         * Do NOT dereference this ptr on CPU if using GPU memory.
         * @tparam T
         * @return
         */
        template<typename T>
        T *data();

        /**
         * Get Reference To Physical Storage In The Heap.
         * Do NOT dereference this ptr on CPU if using GPU memory.
         * @tparam T
         * @return
         */
        template<typename T>
        const T *data() const;

        /**
         *
         * @tparam T
         * @param i
         * @return
         */
        template<typename T>
        T get(size_t i) const;

        /**
         *
         * @tparam T
         * @param i
         * @param val
         */
        template<typename T>
        void set(size_t i, T val);

    private:
        /**
         *
         */
        void free_memory();

    public:
///////////////////////////////////////////////////////////////////
/////////////////***********************************///////////////
/////////////////**  Storage Class Encapsulation  **///////////////
/////////////////***********************************///////////////
///////////////////////////////////////////////////////////////////
        ;

        /**
         *
         * @return
         */
        [[nodiscard]] uint64_t size() const;

        /**
         *
         * @return
         */
        [[nodiscard]] uint64_t capacity() const;

        /**
         *
         * @return
         */
        [[nodiscard]] uint64_t get_item_size() const;

        [[nodiscard]] Device get_device() const;

        /**
         * Copying storage directly will be heavy. Storage ptr can handle it.
         */
        Storage(const Storage &) = delete;


        /**
         * Copying storage directly will be heavy. Storage ptr can handle it.
         */
        Storage &operator=(const Storage &) = delete;

    };

}

#include "temp/Storage.tpp"