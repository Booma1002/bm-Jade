#pragma once
#include <cstdint>
#include "Logger.hpp"
namespace zeza {
    static constexpr int OPER_MAX_DIMS = 16;
    static constexpr int OPER_MAX_OPERANDS = 3;
    constexpr uint64_t MAX_OP_METHODS = 1024;

    //////////////////////////////////////////////////////////
    /////////////////**************************///////////////
    /////////////////**  Device Declaration  **///////////////
    /////////////////**************************///////////////
    //////////////////////////////////////////////////////////
    ;
    /**
     * @brief Specifies the hardware being fetched by the `Tile`
     * @Switching Use `CPU` to select the central processing unit.
     * @Switching Use `CUDA` to select the graphics processing unit.
     * @usage To register a new hardware, manually add it to the enum class 'Device',
     * and specify its unique code.
     * @usage Preferably select a sequential code and write it down in 'hex8' format.
     */
    enum class Device : uint8_t {
        CPU          = 0X1,
        CUDA         = 0X2,
        MAX_DEVICES  = 0X3
    };

    //////////////////////////////////////////////////////////
    /////////////////**************************///////////////
    /////////////////**  OpCode Declaration  **///////////////
    /////////////////**************************///////////////
    //////////////////////////////////////////////////////////
    ;
    /**
     * @brief Specifies the operation being dispatched by the 'Dispatcher'
     * @usage To create a new OpCode, manually add it to the enum class 'Opcode',
     * and specify its unique code.
     * @usage Preferably select a sequential code and write it down in 'hex32' format.
     */
    enum class OpCode : uint32_t {
        NONE      = 0X00,
        SUB       = 0X01,
        MUL       = 0X02,
        COPY      = 0X03,
        MATMUL    = 0X04,
        FILL      = 0X05,
        ADD       = 0X06,
        SIN       = 0X07,
        COS       = 0X08,
        TAN       = 0X09,
        EXP       = 0X0A,
        LOG       = 0X0B,
        CLIP      = 0X0C,

        MAX_OPS    = 0XFF
    };

    enum class DType : uint16_t {
        NONE        = 0X0,
        FLOAT32     = 0X1,
        FLOAT64     = 0X2,
        INT32       = 0X3,
        INT16       = 0X4,
        UINT8       = 0X5,
        UINT16      = 0X6,
        UINT32      = 0X7,
        INT64       = 0X8,
        UINT64      = 0X9,

        MAX_DTYPES = 0XF
    };

    /**
     *
     * @param type Data type
     * @return size in bytes.
     */
    constexpr uint64_t get_dtype_size(DType type) {
        switch(type) {
            case DType::FLOAT64 : return 8;
            case DType::FLOAT32 : return 4;
            case DType::INT64   : return 8;
            case DType::INT32   : return 4;
            case DType::INT16   : return 2;
            case DType::UINT64  : return 8;
            case DType::UINT32  : return 4;
            case DType::UINT16  : return 2;
            case DType::UINT8   : return 1;
            default: return 0;
        }
    }
}// namespace zeza