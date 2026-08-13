#pragma once

#include <concepts>
#include <cstdint>

// Architecture-specific includes and definitions
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <xmmintrin.h>
    #define MPFX_ARCH_X86
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define MPFX_ARCH_ARM64
#else
    #include <cfenv>
    #define MPFX_ARCH_GENERIC
#endif

namespace mpfx {
namespace arch {

#ifdef MPFX_ARCH_X86
    // default MXCSR value with all exceptions masked
    static constexpr unsigned int MXCSR_DEFAULT = 0x1F80;

    // Rounding mode constants for x86 SSE
    static constexpr int RM_RNE = 0x0;  // Round to Nearest Even
    static constexpr int RM_RTN = 0x1;  // Round Toward Negative
    static constexpr int RM_RTP = 0x2;  // Round Toward Positive  
    static constexpr int RM_RTZ = 0x3;  // Round Toward Zero

    // Exception flag constants
    static constexpr unsigned int EXCEPT_INVALID = 0x01;
    static constexpr unsigned int EXCEPT_DENORM = 0x02;
    static constexpr unsigned int EXCEPT_DIVZERO = 0x04;
    static constexpr unsigned int EXCEPT_OVERFLOW = 0x08;
    static constexpr unsigned int EXCEPT_UNDERFLOW = 0x10;
    static constexpr unsigned int EXCEPT_INEXACT = 0x20;
    static constexpr unsigned int EXCEPT_ALL = 0x3F;

    /// @brief Get the current floating-point status and control register.
    /// @return Architecture-specific floating-point status register value.
    inline unsigned int get_fpscr() {
        return _mm_getcsr();
    }

    /// @brief Set the floating-point status and control register.
    /// @param csr Architecture-specific floating-point status register value to set.
    inline void set_fpscr(unsigned int csr) {
        _mm_setcsr(csr);
    }

    /// @brief Clear floating-point exception flags.
    inline void clear_exceptions() {
        unsigned int csr = get_fpscr();
        set_fpscr(csr & ~0x3F); // Clear exception flags (bits 0-5)
    }

    /// @brief Get the current floating-point exception flags.
    /// @return Current exception flags.
    inline unsigned int get_exceptions() {
        return get_fpscr() & EXCEPT_ALL; // Extract exception flags (bits 0-5)
    }

    /// @brief Get the current rounding mode.
    /// @return Current rounding mode value.
    inline int get_rounding_mode() {
        return (get_fpscr() >> 13) & 0x3; // Extract bits 13-14
    }
    
    /// @brief Set the rounding mode.
    /// @param mode New rounding mode value.
    inline void set_rounding_mode(int mode) {
        unsigned int csr = get_fpscr();
        csr = (csr & ~0x6000) | ((mode & 0x3) << 13); // Clear and set bits 13-14
        set_fpscr(csr);
    }
    
    /// @brief Prepares for a round-to-odd operation by setting
    /// the rounding mode to RTZ and clearing exceptions.
    /// @return Previous rounding mode.
    inline unsigned int prepare_rto() {
        static constexpr unsigned int rtz_csr = MXCSR_DEFAULT | (RM_RTZ << 13);
        unsigned int csr = get_fpscr();
        set_fpscr(rtz_csr); // set new CSR with RTZ and cleared exceptions
        return csr;
    }

    /// @brief Retrieve exception status to finalize a round-to-odd operation.
    /// @param old_mode Previous rounding mode.
    /// @return Exception flags.
    inline unsigned int rto_status(unsigned int old_csr) {
        unsigned int csr = get_fpscr();
        const unsigned int exceptions = csr & EXCEPT_ALL; // extract exception flags
        set_fpscr(old_csr & ~EXCEPT_ALL); // clear exception flags
        return exceptions;
    }

#elif defined(MPFX_ARCH_ARM64)
    // Rounding mode constants for ARM64
    static constexpr int RM_RNE = 0x0;  // Round to Nearest Even
    static constexpr int RM_RTP = 0x1;  // Round Toward Positive
    static constexpr int RM_RTN = 0x2;  // Round Toward Negative
    static constexpr int RM_RTZ = 0x3;  // Round Toward Zero
    
    // Exception flag constants for ARM64
    static constexpr unsigned int EXCEPT_INVALID = 0x01;
    static constexpr unsigned int EXCEPT_DIVZERO = 0x02;
    static constexpr unsigned int EXCEPT_OVERFLOW = 0x04;
    static constexpr unsigned int EXCEPT_UNDERFLOW = 0x08;
    static constexpr unsigned int EXCEPT_INEXACT = 0x10;

    // FPCR and FPSR are 64-bit system registers and `mrs`/`msr` take an X
    // register, so the asm operands below must be 64-bit wide to match.

    /// @brief Get the current floating-point status and control register.
    /// @return Architecture-specific floating-point status register value.
    inline unsigned int get_fpscr() {
        uint64_t fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        return static_cast<unsigned int>(fpcr);
    }

    /// @brief Set the floating-point status and control register.
    /// @param csr Architecture-specific floating-point status register value to set.
    inline void set_fpscr(unsigned int csr) {
        const uint64_t fpcr = csr;
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    }

    /// @brief Clear floating-point exception flags.
    inline void clear_exceptions() {
        __asm__ volatile("msr fpsr, xzr");
    }

    /// @brief Get the current floating-point exception flags.
    /// @return Current exception flags.
    inline unsigned int get_exceptions() {
        uint64_t fpsr;
        __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr));
        return static_cast<unsigned int>(fpsr) & 0x1F; // Extract exception flags (bits 0-4)
    }

    /// @brief Get the current rounding mode.
    /// @return Current rounding mode value.
    inline int get_rounding_mode() {
        return (get_fpscr() >> 22) & 0x3; // Extract RMode bits 23-22
    }
    
    /// @brief Set the rounding mode.
    /// @param mode New rounding mode value.
    inline void set_rounding_mode(int mode) {
        unsigned int fpcr = get_fpscr();
        fpcr = (fpcr & ~0xC00000) | ((mode & 0x3) << 22); // Clear and set bits 23-22
        set_fpscr(fpcr);
    }
    
    /// @brief Prepares for a round-to-odd operation by setting
    /// the rounding mode to RTZ and clearing exceptions.
    /// @return Previous rounding mode.
    inline int prepare_rto() {
        uint64_t fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        const int old_mode = (fpcr >> 22) & 0x3;        // Extract old rounding mode
        fpcr = (fpcr & ~0xC00000ull) | (0x3ull << 22);  // Set RTZ mode
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
        __asm__ volatile("msr fpsr, xzr"); // Clear exceptions
        return old_mode;
    }

    /// @brief Retrieve exception status to finalize a round-to-odd operation.
    /// @param old_mode Previous rounding mode.
    /// @return Exception flags.
    inline int rto_status(int old_mode) {
        uint64_t fpsr, fpcr;
        __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr));
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        const int exceptions = fpsr & 0x1E;       // Extract overflow, underflow, inexact flags
        fpcr = (fpcr & ~0xC00000ull) | (static_cast<uint64_t>(old_mode & 0x3) << 22);  // Restore rounding mode
        __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
        return exceptions;
    }
    
#else

    // Rounding mode constants for generic implementation
    static constexpr int RM_RNE = FE_TONEAREST;   // Round to Nearest Even
    static constexpr int RM_RTN = FE_DOWNWARD;    // Round Toward Negative
    static constexpr int RM_RTP = FE_UPWARD;      // Round Toward Positive
    static constexpr int RM_RTZ = FE_TOWARDZERO;  // Round Toward Zero
    
    // Use standard exception constants
    static constexpr int EXCEPT_INVALID = FE_INVALID;
    static constexpr int EXCEPT_DIVZERO = FE_DIVBYZERO;
    static constexpr int EXCEPT_OVERFLOW = FE_OVERFLOW;
    static constexpr int EXCEPT_UNDERFLOW = FE_UNDERFLOW;
    static constexpr int EXCEPT_INEXACT = FE_INEXACT;

    /// @brief Get the current floating-point status and control register.
    /// @return Architecture-specific floating-point status register value.
    inline int get_fpscr() {
        return std::fegetround();
    }
    
    /// @brief Set the floating-point status and control register.
    /// @param rm Architecture-specific floating-point status register value to set.
    inline void set_fpscr(int rm) {
        std::fesetround(rm);
    }
    
    /// @brief Clear floating-point exception flags.
    inline void clear_exceptions() {
        std::feclearexcept(FE_ALL_EXCEPT);
    }

    /// @brief Get the current floating-point exception flags.
    /// @return Current exception flags.
    inline int get_exceptions() {
        return std::fetestexcept(FE_ALL_EXCEPT);
    }

    /// @brief Get the current rounding mode.
    /// @return Current rounding mode value.
    inline int get_rounding_mode() {
        return std::fegetround();
    }
    
    /// @brief Set the rounding mode.
    /// @param mode New rounding mode value.
    inline void set_rounding_mode(int mode) {
        std::fesetround(mode);
    }
    
    /// @brief Prepares for a round-to-odd operation by setting
    /// the rounding mode to RTZ and clearing exceptions.
    /// @return Previous rounding mode.
    inline int prepare_rto() {
        const int old_mode = std::fegetround();
        std::fesetround(RM_RTZ);
        std::feclearexcept(FE_ALL_EXCEPT);
        return old_mode;
    }

    /// @brief Retrieve exception status to finalize a round-to-odd operation.
    /// @param old_mode Previous rounding mode.
    /// @return Exception flags.
    inline int rto_status(int old_mode) {
        const int exceptions = std::fetestexcept(FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
        std::fesetround(old_mode);
        return exceptions;
    }

#endif

/// Tells the compiler that the enclosing block inspects the floating-point
/// environment, so it may not assume the default rounding mode or that the status
/// flags go unread. Clang honors this by emitting constrained FP operations, which
/// carry side effects and cannot drift across the register accesses. GCC does not
/// implement the pragma (it warns and ignores it), so `fp_barrier` below remains
/// the only protection there.
#if defined(__clang__)
    #define MPFX_FENV_ACCESS_ON _Pragma("STDC FENV_ACCESS ON")
#else
    #define MPFX_FENV_ACCESS_ON
#endif

/// @brief Forces `v` to be materialized here, ordering its computation against
/// neighboring `asm volatile` statements.
///
/// Floating-point arithmetic carries no dependency on the register accesses that
/// bracket a round-to-odd operation, and `FENV_ACCESS` is off by default, so the
/// compiler is otherwise free to schedule the operation outside that window and
/// test status flags it never set.
template <std::floating_point T>
inline void fp_barrier(T& v) {
#if defined(MPFX_ARCH_X86)
    __asm__ volatile("" : "+x"(v));
#elif defined(MPFX_ARCH_ARM64)
    __asm__ volatile("" : "+w"(v));
#else
    volatile T tmp = v;
    v = tmp;
#endif
}

} // end namespace arch
} // end namespace mpfx
