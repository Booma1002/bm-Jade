#pragma once
#include "Registry.hpp"
namespace zeza {
////////////////////////////////////////////////////////////////////////
/////////////////***************************************////////////////
/////////////////**  Dispatcher Class Initialization  **////////////////
/////////////////***************************************////////////////
////////////////////////////////////////////////////////////////////////
    ;

    class Tile;

    struct Dispatcher {
/////////////////////////////////////////////////////////////
/////////////////****************************////////////////
/////////////////**  Dispatcher Executors  **////////////////
/////////////////****************************////////////////
/////////////////////////////////////////////////////////////
        /**
         *
         * @param op
         * @param out
         * @param a
         * @param b
         */
        static void execute_binary(OpCode op, Tile &out, const Tile &a, const Tile &b);

        /**
         *
         * @param op
         * @param out
         * @param a
         */
        static void
        execute_unary(OpCode op, Tile &out, const Tile &a, const double left = 0.f, const double right = 0.f);

        static void execute_scalar(OpCode op, Tile &out, double a);
    };

}// namespace zeza

#include "temp/Dispatcher.tpp"
