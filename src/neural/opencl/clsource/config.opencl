// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

typedef float net_t;
#define vload_net_t(offset,p) ((p)[(offset)])
#define vstore_net_t(data,offset,p) (((p)[(offset)])=(data))

#define BOARD_SIZE 8
#define BOARD_SQUARES (BOARD_SIZE*BOARD_SIZE)

// End of the C++11 raw string literal
)"
