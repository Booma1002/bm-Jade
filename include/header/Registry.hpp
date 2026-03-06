#pragma once
#include "Enums.hpp"
#include <cstdint>
#include <stdexcept>
#include <iostream>
namespace zeza {
    struct TileOperator;

/**
 * @brief Pointer to a function (i.e. Kernel).
 * @accepts accepts a reference to a 'TileOperator' object.
 * @returns returns void;
 */
    using Kernel = void (*)(TileOperator &);

    class RegistryException : public std::exception {
        std::string msg;
    public:
        explicit RegistryException(std::string message) : msg(std::move(message)) {}
        virtual const char *what() const noexcept override { return msg.c_str(); }
    };

/////////////////////////////////////////////////////////////////////
/////////////////*************************************///////////////
/////////////////**  Registry Class Initialization  **///////////////
/////////////////*************************************///////////////
/////////////////////////////////////////////////////////////////////
    ;

/**
 *
 */
    class Registry {
////////////////////////////////////////////////////////////
/////////////////****************************///////////////
/////////////////**  Registry Class Setup  **///////////////
/////////////////****************************///////////////
////////////////////////////////////////////////////////////
    public:
        Kernel kernel_table[static_cast<int>(OpCode::MAX_OPS)][static_cast<int>(Device::MAX_DEVICES)]{};

        /**
         *
         */
        Registry();

        /**
         * @note No need to re-assign registry
         */
        Registry(const Registry &) = delete;

        /**
         * @note No need to re-assign registry
         */
        Registry &operator=(const Registry &) = delete;

////////////////////////////////////////////////////////////////
/////////////////********************************///////////////
/////////////////**  Registry Class Utilities  **///////////////
/////////////////********************************///////////////
////////////////////////////////////////////////////////////////
        /**
         *
         * @return
         */
        static Registry &get();

        /**
         *
         * @param op
         * @param dev
         * @param func
         */
        void register_kernel(OpCode op, Device dev, Kernel func);

        /**
         *
         * @param op
         * @param dev
         * @return
         */
        Kernel lookup(OpCode op, Device dev);
    };

//////////////////////////////////////////////////////////////
/////////////////******************************///////////////
/////////////////**  Kernel Registry Portal  **///////////////
/////////////////******************************///////////////
//////////////////////////////////////////////////////////////
    ;
/**
 * @brief Register specified `OpCode` at the selected `Device`,
 * and uses `Registry` class static constructor instantiation
 * to register the specified `Kernel` at compile time
 * @param OP_ENUM OpCode that is being tied.
 * @param DEV_ENUM Device number that is being used.
 * @param FUNC_PTR Kernel being tied to.
 * @returns True if _reg_Dev_Op is successfully registered.
 * @example
 * @code
 *  void cpu_add_kernel(TileOperator& op) {
 *      ...
 *  }
 *  REGISTER_KERNEL(ADD, CPU, cpu_add_kernel);
 *
 */
#define REGISTER_KERNEL(OP_ENUM, DEV_ENUM, FUNC_PTR) \
    static bool _reg_##DEV_ENUM##_##OP_ENUM = []() { \
        Registry::get().register_kernel(OpCode::OP_ENUM, Device::DEV_ENUM, FUNC_PTR); \
        return true; \
    }();
    ;;;;;;;;;;

}// namespace zeza