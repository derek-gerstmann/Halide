#include <algorithm>
#include <sstream>

#include "CodeGen_GPU_Dev.h"
#include "CodeGen_Internal.h"
#include "CodeGen_Vulkan_Dev.h"
#include "Debug.h"
#include "Deinterleave.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Scope.h"
#include "Target.h"
#include "SPIRV_IR.h"

// Temporary:
#include <fstream>

namespace Halide {
namespace Internal {

class CodeGen_LLVM;

namespace {  // anonymous

// --

template<typename CodeGenT, typename ValueT>
ValueT lower_int_uint_div(CodeGenT *cg, Expr a, Expr b);

template<typename CodeGenT, typename ValueT>
ValueT lower_int_uint_mod(CodeGenT *cg, Expr a, Expr b);

class CodeGen_Vulkan_Dev : public CodeGen_GPU_Dev {
public:
    CodeGen_Vulkan_Dev(Target target);

    /** Compile a GPU kernel into the module. This may be called many times
     * with different kernels, which will all be accumulated into a single
     * source module shared by a given Halide pipeline. */
    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args) override;

    /** (Re)initialize the GPU kernel module. This is separate from compile,
     * since a GPU device module will often have many kernels compiled into it
     * for a single pipeline. */
    void init_module() override;

    std::vector<char> compile_to_src() override;

    std::string get_current_kernel_name() override;

    void dump() override;

    std::string print_gpu_name(const std::string &name) override;

    std::string api_unique_name() override {
        return "vulkan";
    }

protected:
    class SPIRVEmitter : public IRVisitor {

    public:
        SPIRVEmitter() = default;

        using IRVisitor::visit;

        void visit(const Variable *) override;
        void visit(const IntImm *) override;
        void visit(const UIntImm *) override;
        void visit(const StringImm *) override;
        void visit(const FloatImm *) override;
        void visit(const Cast *) override;
        void visit(const Add *) override;
        void visit(const Sub *) override;
        void visit(const Mul *) override;
        void visit(const Div *) override;
        void visit(const Mod *) override;
        void visit(const Max *) override;
        void visit(const Min *) override;
        void visit(const EQ *) override;
        void visit(const NE *) override;
        void visit(const LT *) override;
        void visit(const LE *) override;
        void visit(const GT *) override;
        void visit(const GE *) override;
        void visit(const And *) override;
        void visit(const Or *) override;
        void visit(const Not *) override;
        void visit(const Call *) override;
        void visit(const Select *) override;
        void visit(const Load *) override;
        void visit(const Store *) override;
        void visit(const Let *) override;
        void visit(const LetStmt *) override;
        void visit(const AssertStmt *) override;
        void visit(const ProducerConsumer *) override;
        void visit(const For *) override;
        void visit(const Ramp *) override;
        void visit(const Broadcast *) override;
        void visit(const Provide *) override;
        void visit(const Allocate *) override;
        void visit(const Free *) override;
        void visit(const Realize *) override;
        void visit(const IfThenElse *) override;
        void visit(const Evaluate *) override;
        void visit(const Shuffle *) override;
        void visit(const Prefetch *) override;
        void visit(const Fork *) override;
        void visit(const Acquire *) override;

        void visit_binop(Type t, const Expr &a, const Expr &b, uint32_t opcode);

        SpvContext context;
        
        // ID of last generated Expr.
        uint32_t id;
        // IDs are allocated in numerical order of use.
        uint32_t next_id{0};

        // The void type does not map to a Halide type, but must be unique
        uint32_t void_id;

        // SPIR-V instructions in a module must be in a specific
        // order. This order doesn't correspond to the order in which they
        // are created. Hence we generate into a set of blocks, each of
        // which is added to at its end. In compile_to_src, these are
        // concatenated to form a complete SPIR-V module.  We also
        // represent the temporaries as vectors of uint32_t rather than
        // char for ease of adding words to them.
        std::vector<uint32_t> spir_v_header;
        std::vector<uint32_t> spir_v_entrypoints;
        std::vector<uint32_t> spir_v_execution_modes;
        std::vector<uint32_t> spir_v_annotations;
        std::vector<uint32_t> spir_v_types;
        std::vector<uint32_t> spir_v_kernels;
        // The next one is cleared in between kernels, and tracks the allocations
        std::vector<uint32_t> spir_v_kernel_allocations;

        // Id of entry point for kernel currently being compiled.
        uint32_t current_function_id;

        // Top-level function for adding kernels
        void add_kernel(const Stmt &s, const std::string &name, const std::vector<DeviceArgument> &args);

        // Function for allocating variables in function scope, with optional initializer.
        // These will appear at the beginning of the function, as required by SPIR-V
        void add_allocation(uint32_t result_type_id, uint32_t result_id, uint32_t storage_class, uint32_t initializer = 0);

        std::map<Type, uint32_t> type_map;
        std::map<std::pair<Type, uint32_t>, uint32_t> pointer_type_map;
        std::map<Type, uint32_t> pair_type_map;
        std::map<std::string, uint32_t> constant_map;

        void add_instruction(std::vector<uint32_t> &region, uint32_t opcode,
                             std::initializer_list<uint32_t> words);
        void add_instruction(uint32_t opcode, std::initializer_list<uint32_t> words);
        void add_instruction(std::vector<uint32_t> &region, uint32_t opcode,
                             std::vector<uint32_t> words);
        void add_instruction(uint32_t opcode, std::vector<uint32_t> words);
        uint32_t map_type(const Type &type);
        uint32_t map_pointer_type(const Type &type, uint32_t storage_class);
        uint32_t map_type_to_pair(const Type &t);
        uint32_t emit_constant(const Type &t, const void *data);
        void scalarize(const Expr &e);

        // The scope contains both the symbol and its storage class
        Scope<std::pair<uint32_t, uint32_t>> symbol_table;

        // The workgroup size.  Must be the same for all kernels.
        uint32_t workgroup_size[3];

        struct PhiNodeInputs {
            uint32_t ids[4];
        };
        // Returns Phi node inputs.
        template<typename StmtOrExpr>
        PhiNodeInputs emit_if_then_else(const Expr &condition, StmtOrExpr then_case, StmtOrExpr else_case);
    } emitter;

    std::string current_kernel_name;
};

// --

void CodeGen_Vulkan_Dev::SPIRVEmitter::add_instruction(std::vector<uint32_t> &region, uint32_t opcode,
                                                       std::initializer_list<uint32_t> words) {
    region.push_back(((1 + words.size()) << 16) | opcode);
    region.insert(region.end(), words.begin(), words.end());
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::add_instruction(uint32_t opcode, std::initializer_list<uint32_t> words) {
    spir_v_kernels.push_back(((1 + words.size()) << 16) | opcode);
    spir_v_kernels.insert(spir_v_kernels.end(), words.begin(), words.end());
}
void CodeGen_Vulkan_Dev::SPIRVEmitter::add_instruction(std::vector<uint32_t> &region, uint32_t opcode,
                                                       std::vector<uint32_t> words) {
    region.push_back(((1 + words.size()) << 16) | opcode);
    region.insert(region.end(), words.begin(), words.end());
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::add_instruction(uint32_t opcode, std::vector<uint32_t> words) {
    spir_v_kernels.push_back(((1 + words.size()) << 16) | opcode);
    spir_v_kernels.insert(spir_v_kernels.end(), words.begin(), words.end());
}

uint32_t CodeGen_Vulkan_Dev::SPIRVEmitter::emit_constant(const Type &t, const void *data) {
    // TODO: this needs to emit OpConstantComposite for constants with lane > 1
    std::string key(t.bytes() + 4, ' ');
    key[0] = t.code();
    key[1] = t.bits();
    key[2] = t.lanes() & 0xff;
    key[3] = (t.lanes() >> 8) & 0xff;
    const char *data_char = (const char *)data;
    for (int i = 0; i < t.bytes(); i++) {
        key[i + 4] = data_char[i];
    }

    debug(3) << "emit_constant for type " << t << "\n";
    auto item = constant_map.find(key);
    if (item == constant_map.end()) {
        uint32_t type_id = map_type(t);
        uint32_t extra_words = (t.bytes() + 3) / 4;
        uint32_t constant_id = next_id++;
        spir_v_types.push_back(((3 + extra_words) << 16) | SpvOpConstant);
        spir_v_types.push_back(type_id);
        spir_v_types.push_back(constant_id);

        const uint8_t *data_temp = (const uint8_t *)data;
        size_t bytes_copied = 0;
        for (uint32_t i = 0; i < extra_words; i++) {
            uint32_t word;
            size_t to_copy = std::min(t.bytes() - bytes_copied, (size_t)4);
            memcpy(&word, data_temp, to_copy);
            bytes_copied += to_copy;
            spir_v_types.push_back(word);
            data_temp++;
        }
        return constant_id;
    } else {
        return item->second;
    }
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::scalarize(const Expr &e) {
    internal_assert(e.type().is_vector()) << "CodeGen_Vulkan_Dev::SPIRVEmitter::scalarize must be called with an expression of vector type.\n";
    uint32_t type_id = map_type(e.type());

    uint32_t result_id = next_id++;
    add_instruction(SpvOpConstantNull, {type_id, result_id});

    for (int i = 0; i < e.type().lanes(); i++) {
        extract_lane(e, i).accept(this);
        uint32_t composite_vec = next_id++;
        add_instruction(SpvOpVectorInsertDynamic, {type_id, composite_vec, (uint32_t)i, result_id, id});
        result_id = composite_vec;
    }
    id = result_id;
}

uint32_t CodeGen_Vulkan_Dev::SPIRVEmitter::map_type(const Type &t) {
    auto key_typecode = t.code();

    Type t_key(key_typecode, t.bits(), t.lanes());

    auto item = type_map.find(t_key);
    if (item == type_map.end()) {
        // TODO, handle arrays, pointers, halide_buffer_t
        uint32_t type_id = 0;
        if (t.lanes() != 1) {
            uint32_t base_id = map_type(t.with_lanes(1));
            type_id = next_id++;
            add_instruction(spir_v_types, SpvOpTypeVector, {type_id, base_id, (uint32_t)t.lanes()});
        } else {
            if (t.is_float()) {
                type_id = next_id++;
                add_instruction(spir_v_types, SpvOpTypeFloat, {type_id, (uint32_t)t.bits()});
            } else if (t.is_bool()) {
                type_id = next_id++;
                add_instruction(spir_v_types, SpvOpTypeBool, {type_id});
            } else if (t.is_int_or_uint()) {
                type_id = next_id++;
                uint32_t signedness = t.is_uint() ? 0 : 1;
                add_instruction(spir_v_types, SpvOpTypeInt, {type_id, (uint32_t)t.bits(), signedness});
            } else {
                internal_error << "Unsupported type in Vulkan backend " << t << "\n";
            }
        }
        type_map[t_key] = type_id;
        return type_id;
    } else {
        return item->second;
    }
}

uint32_t CodeGen_Vulkan_Dev::SPIRVEmitter::map_type_to_pair(const Type &t) {
    uint32_t &ref = pair_type_map[t];

    if (ref == 0) {
        uint32_t base_type = map_type(t);

        uint32_t type_id = next_id++;

        add_instruction(spir_v_types, SpvOpTypeStruct, {type_id, base_type, base_type});
        ref = type_id;
    }
    return ref;
}

uint32_t CodeGen_Vulkan_Dev::SPIRVEmitter::map_pointer_type(const Type &type, const uint32_t storage_class) {
    auto key = std::make_pair(type, storage_class);
    uint32_t &ref = pointer_type_map[key];
    if (ref == 0) {
        uint32_t base_type_id = map_type(type);
        ref = next_id++;
        add_instruction(spir_v_types, SpvOpTypePointer, {ref, storage_class, base_type_id});
        pointer_type_map[key] = ref;
    }

    return ref;
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Variable *var) {
    id = symbol_table.get(var->name).first;
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const IntImm *imm) {
    id = emit_constant(imm->type, &imm->value);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const UIntImm *imm) {
    id = emit_constant(imm->type, &imm->value);
}

namespace {
void encode_string(std::vector<uint32_t> &section, const uint32_t words,
                   const size_t str_size, const char *str) {
    size_t bytes_copied = 0;
    for (uint32_t i = 0; i < words; i++) {
        uint32_t word;
        size_t to_copy = std::min(str_size + 1 - bytes_copied, (size_t)4);
        memcpy(&word, str, to_copy);
        bytes_copied += to_copy;
        section.push_back(word);
        str += 4;
    }
}
}  // namespace
void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const StringImm *imm) {
    uint32_t extra_words = (imm->value.size() + 1 + 3) / 4;
    id = next_id++;
    spir_v_kernels.push_back(((2 + extra_words) << 16) | SpvOpString);
    spir_v_kernels.push_back(id);

    const char *data_temp = (const char *)imm->value.c_str();
    const size_t data_size = imm->value.size();
    encode_string(spir_v_kernels, extra_words, data_size, data_temp);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const FloatImm *imm) {
    user_assert(imm->type.bits() == 32) << "Vulkan backend currently only supports 32-bit floats\n";
    float float_val = (float)(imm->value);
    id = emit_constant(imm->type, &float_val);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Cast *op) {
    uint32_t opcode = 0;
    if (op->value.type().is_float()) {
        if (op->type.is_float()) {
            opcode = SpvOpFConvert;
        } else if (op->type.is_uint()) {
            opcode = SpvOpConvertFToU;
        } else if (op->type.is_int()) {
            opcode = SpvOpConvertFToS;
        } else {
            internal_error << "Vulkan cast unhandled case " << op->value.type() << " to " << op->type << "\n";
        }
    } else if (op->value.type().is_uint()) {
        if (op->type.is_float()) {
            opcode = SpvOpConvertUToF;
        } else if (op->type.is_uint()) {
            opcode = SpvOpUConvert;
        } else if (op->type.is_int()) {
            opcode = SpvOpSatConvertUToS;
        } else {
            internal_error << "Vulkan cast unhandled case " << op->value.type() << " to " << op->type << "\n";
        }
    } else if (op->value.type().is_int()) {
        if (op->type.is_float()) {
            opcode = SpvOpConvertSToF;
        } else if (op->type.is_uint()) {
            opcode = SpvOpSatConvertSToU;
        } else if (op->type.is_int()) {
            opcode = SpvOpSConvert;
        } else {
            internal_error << "Vulkan cast unhandled case " << op->value.type() << " to " << op->type << "\n";
        }
    } else {
        internal_error << "Vulkan cast unhandled case " << op->value.type() << " to " << op->type << "\n";
    }

    uint32_t type_id = map_type(op->type);
    op->value.accept(this);
    uint32_t src_id = id;
    id = next_id++;
    add_instruction(opcode, {type_id, id, src_id});
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Add *op) {
    visit_binop(op->type, op->a, op->b, op->type.is_float() ? SpvOpFAdd : SpvOpIAdd);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Sub *op) {
    visit_binop(op->type, op->a, op->b, op->type.is_float() ? SpvOpFSub : SpvOpISub);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Mul *op) {
    visit_binop(op->type, op->a, op->b, op->type.is_float() ? SpvOpFMul : SpvOpIMul);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Div *op) {
    user_assert(!is_const_zero(op->b)) << "Division by constant zero in expression: " << Expr(op) << "\n";

    if (op->type.is_float()) {
        visit_binop(op->type, op->a, op->b, SpvOpFDiv);
    } else {
        Expr e = lower_int_uint_div(op->a, op->b);
        e.accept(this);
    }
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Mod *op) {
    if (op->type.is_float()) {
        // Takes sign of result from op->b
        visit_binop(op->type, op->a, op->b, SpvOpFMod);
    } else {
        Expr e = lower_int_uint_mod(op->a, op->b);
        e.accept(this);
    }
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Max *op) {
    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);
    Expr temp = Let::make(a_name, op->a,
                          Let::make(b_name, op->b, select(a > b, a, b)));
    temp.accept(this);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Min *op) {
    std::string a_name = unique_name('a');
    std::string b_name = unique_name('b');
    Expr a = Variable::make(op->a.type(), a_name);
    Expr b = Variable::make(op->b.type(), b_name);
    Expr temp = Let::make(a_name, op->a,
                          Let::make(b_name, op->b, select(a < b, a, b)));
    temp.accept(this);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const EQ *op) {
    visit_binop(op->type, op->a, op->b, op->type.is_float() ? SpvOpFOrdEqual : SpvOpIEqual);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const NE *op) {
    visit_binop(op->type, op->a, op->b, op->type.is_float() ? SpvOpFOrdNotEqual : SpvOpINotEqual);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const LT *op) {
    uint32_t opcode = 0;
    if (op->a.type().is_float()) {
        opcode = SpvOpFOrdLessThan;
    } else if (op->a.type().is_int()) {
        opcode = SpvOpSLessThan;
    } else if (op->a.type().is_uint()) {
        opcode = SpvOpULessThan;
    } else {
        internal_error << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const LT *op): unhandled type: " << op->a.type() << "\n";
    }
    visit_binop(op->type, op->a, op->b, opcode);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const LE *op) {
    uint32_t opcode = 0;
    if (op->a.type().is_float()) {
        opcode = SpvOpFOrdLessThanEqual;
    } else if (op->a.type().is_int()) {
        opcode = SpvOpSLessThanEqual;
    } else if (op->a.type().is_uint()) {
        opcode = SpvOpULessThanEqual;
    } else {
        internal_error << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const LE *op): unhandled type: " << op->a.type() << "\n";
    }
    visit_binop(op->type, op->a, op->b, opcode);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const GT *op) {
    uint32_t opcode = 0;
    if (op->a.type().is_float()) {
        opcode = SpvOpFOrdGreaterThan;
    } else if (op->a.type().is_int()) {
        opcode = SpvOpSGreaterThan;
    } else if (op->a.type().is_uint()) {
        opcode = SpvOpUGreaterThan;
    } else {
        internal_error << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const GT *op): unhandled type: " << op->a.type() << "\n";
    }
    visit_binop(op->type, op->a, op->b, opcode);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const GE *op) {
    uint32_t opcode = 0;
    if (op->a.type().is_float()) {
        opcode = SpvOpFOrdGreaterThanEqual;
    } else if (op->a.type().is_int()) {
        opcode = SpvOpSGreaterThanEqual;
    } else if (op->a.type().is_uint()) {
        opcode = SpvOpUGreaterThanEqual;
    } else {
        internal_error << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const GE *op): unhandled type: " << op->a.type() << "\n";
    }
    visit_binop(op->type, op->a, op->b, opcode);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const And *op) {
    visit_binop(op->type, op->a, op->b, SpvOpLogicalAnd);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Or *op) {
    visit_binop(op->type, op->a, op->b, SpvOpLogicalOr);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Not *op) {
    uint32_t type_id = map_type(op->type);
    op->a.accept(this);
    uint32_t a_id = id;
    id = next_id++;
    add_instruction(SpvOpLogicalNot, {type_id, id, a_id});
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Call *op) {
    if (op->is_intrinsic(Call::gpu_thread_barrier)) {
        // TODO: Check the scopes here and figure out if this is the
        // right memory barrier. Might be able to use
        // SpvMemorySemanticsMaskNone instead.
        add_instruction(SpvOpControlBarrier, {current_function_id, current_function_id,
                                              SpvMemorySemanticsAcquireReleaseMask});
    } else if (op->is_intrinsic(Call::bitwise_and)) {
        internal_assert(op->args.size() == 2);
        visit_binop(op->type, op->args[0], op->args[1], SpvOpBitwiseAnd);
    } else if (op->is_intrinsic(Call::bitwise_xor)) {
        internal_assert(op->args.size() == 2);
        visit_binop(op->type, op->args[0], op->args[1], SpvOpBitwiseXor);
    } else if (op->is_intrinsic(Call::bitwise_or)) {
        internal_assert(op->args.size() == 2);
        visit_binop(op->type, op->args[0], op->args[1], SpvOpBitwiseOr);
    } else if (op->is_intrinsic(Call::bitwise_not)) {
        internal_assert(op->args.size() == 1);
        uint32_t type_id = map_type(op->type);
        op->args[0]->accept(this);
        uint32_t arg_id = id;
        id = next_id++;
        add_instruction(SpvOpNot, {type_id, id, arg_id});
    } else if (op->is_intrinsic(Call::reinterpret)) {
    } else if (op->is_intrinsic(Call::if_then_else)) {
        if (op->type.is_vector()) {
            scalarize(op);
        } else {
            internal_assert(op->args.size() == 3);
            auto phi_inputs = emit_if_then_else(op->args[0], op->args[1], op->args[2]);
            // Generate Phi node if used as an expression.

            uint32_t type_id = map_type(op->type);
            id = next_id++;
            spir_v_kernels.push_back((7 << 16) | SpvOpPhi);
            spir_v_kernels.push_back(type_id);
            spir_v_kernels.push_back(id);
            spir_v_kernels.insert(spir_v_kernels.end(), phi_inputs.ids, phi_inputs.ids + 4);
        }
    } else if (op->is_intrinsic(Call::IntrinsicOp::div_round_to_zero)) {
        internal_assert(op->args.size() == 2);
        uint32_t opcode = 0;
        if (op->type.is_int()) {
            opcode = SpvOpSDiv;
        } else if (op->type.is_uint()) {
            opcode = SpvOpUDiv;
        } else {
            internal_error << "div_round_to_zero of non-integer type.\n";
        }
        visit_binop(op->type, op->args[0], op->args[1], opcode);
    } else if (op->is_intrinsic(Call::IntrinsicOp::mod_round_to_zero)) {
        internal_assert(op->args.size() == 2);
        uint32_t opcode = 0;
        if (op->type.is_int()) {
            opcode = SpvOpSMod;
        } else if (op->type.is_uint()) {
            opcode = SpvOpUMod;
        } else {
            internal_error << "mod_round_to_zero of non-integer type.\n";
        }
        visit_binop(op->type, op->args[0], op->args[1], opcode);
    } else if (op->is_intrinsic(Call::IntrinsicOp::mul_shift_right)) {
        internal_assert(op->args.size() == 3);
        uint32_t type_id = map_type(op->type);

        op->args[0].accept(this);
        uint32_t a_id = id;
        op->args[1].accept(this);
        uint32_t b_id = id;

        uint32_t pair_type_id = map_type_to_pair(op->type);

        // Double width multiply
        uint32_t product_pair = next_id++;
        spir_v_kernels.push_back((5 << 16) | (op->type.is_uint() ? SpvOpUMulExtended : SpvOpSMulExtended));
        spir_v_kernels.push_back(pair_type_id);
        spir_v_kernels.push_back(a_id);
        spir_v_kernels.push_back(b_id);

        uint32_t high_item_id = next_id++;
        spir_v_kernels.push_back((5 << 16) | SpvOpCompositeExtract);
        spir_v_kernels.push_back(type_id);
        spir_v_kernels.push_back(high_item_id);
        spir_v_kernels.push_back(product_pair);
        spir_v_kernels.push_back(1);

        const UIntImm *shift = op->args[2].as<UIntImm>();
        internal_assert(shift != nullptr) << "Third argument to mul_shift_right intrinsic must be an unsigned integer immediate.\n";

        uint32_t result_id;
        if (shift->value != 0) {
            // TODO: This code depends on compilation happening on a little-endian host.
            uint32_t shr_id = emit_constant(shift->type, &shift->value);
            result_id = next_id++;
            spir_v_kernels.push_back((5 << 16) | (op->type.is_uint() ? SpvOpShiftRightLogical : SpvOpShiftRightArithmetic));
            spir_v_kernels.push_back(type_id);
            spir_v_kernels.push_back(result_id);
            spir_v_kernels.push_back(high_item_id);
            spir_v_kernels.push_back(shr_id);
        } else {
            result_id = high_item_id;
        }

        id = result_id;
    } else if (op->is_intrinsic(Call::IntrinsicOp::sorted_avg)) {
        internal_assert(op->args.size() == 2);
        // b > a, so the following works without widening:
        // a + (b - a)/2
        Expr e = op->args[0] + (op->args[1] - op->args[0]) / 2;
        e.accept(this);
    }
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Select *op) {
    uint32_t type_id = map_type(op->type);
    op->condition.accept(this);
    uint32_t cond_id = id;
    op->true_value.accept(this);
    uint32_t true_id = id;
    op->false_value.accept(this);
    uint32_t false_id = id;
    id = next_id++;
    add_instruction(SpvOpSelect, {type_id, id, cond_id, true_id, false_id});
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Load *op) {
    debug(2) << "Vulkan codegen: Load: " << (Expr)op << "\n";
    user_assert(is_const_one(op->predicate)) << "Predicated loads not supported by the Vulkan backend\n";

    // TODO: implement vector loads
    // TODO: correct casting to the appropriate memory space

//    internal_assert(!(op->index.type().is_vector()));
    internal_assert(op->param.defined() && op->param.is_buffer());

    // Construct the pointer to read from
    auto id_and_storage_class = symbol_table.get(op->name);
    uint32_t base_id = id_and_storage_class.first;
    uint32_t storage_class = id_and_storage_class.second;

    op->index.accept(this);
    uint32_t index_id = id;
    uint32_t ptr_type_id = map_pointer_type(op->type, storage_class);
    uint32_t access_chain_id = next_id++;
    auto zero = 0;
    add_instruction(SpvOpInBoundsAccessChain, {ptr_type_id, access_chain_id, base_id,
                                               emit_constant(UInt(32), &zero), index_id});

    id = next_id++;
    uint32_t result_type_id = map_type(op->type);
    add_instruction(SpvOpLoad, {result_type_id, id, access_chain_id});
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Store *op) {
    debug(2) << "Vulkan codegen: Store: " << (Stmt)op << "\n";

    user_assert(is_const_one(op->predicate)) << "Predicated stores not supported by the Vulkan backend\n";

    // TODO: implement vector writes
    // TODO: correct casting to the appropriate memory space

//     internal_assert(!(op->index.type().is_vector()));
    internal_assert(op->param.defined() && op->param.is_buffer());

    op->value.accept(this);
    uint32_t value_id = id;

    // Construct the pointer to write to
    auto id_and_storage_class = symbol_table.get(op->name);
    uint32_t base_id = id_and_storage_class.first;
    uint32_t storage_class = id_and_storage_class.second;

    op->index.accept(this);
    uint32_t index_id = id;
    uint32_t ptr_type_id = map_pointer_type(op->value.type(), storage_class);
    uint32_t access_chain_id = next_id++;
    auto zero = 0;
    add_instruction(SpvOpInBoundsAccessChain, {ptr_type_id, access_chain_id, base_id,
                                               emit_constant(UInt(32), &zero), index_id});

    add_instruction(SpvOpStore, {access_chain_id, value_id});
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Let *let) {
    let->value.accept(this);
    ScopedBinding<std::pair<uint32_t, uint32_t>> binding(symbol_table, let->name, {id, SpvStorageClassFunction});
    let->body.accept(this);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const LetStmt *let) {
    let->value.accept(this);
    ScopedBinding<std::pair<uint32_t, uint32_t>> binding(symbol_table, let->name, {id, SpvStorageClassFunction});
    let->body.accept(this);
    // TODO: Figure out undef here?
    id = 0xffffffff;
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const AssertStmt *) {
    // TODO: Fill this in.
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const ProducerConsumer *) {
    // I believe these nodes are solely for annotation purposes.
#if 0
    string name;
    if (op->is_producer) {
        name = std::string("produce ") + op->name;
    } else {
        name = std::string("consume ") + op->name;
    }
    BasicBlock *produce = BasicBlock::Create(*context, name, function);
    builder->CreateBr(produce);
    builder->SetInsertPoint(produce);
    codegen(op->body);
#endif
}

namespace {
std::pair<std::string, uint32_t> simt_intrinsic(const std::string &name) {
    if (ends_with(name, ".__thread_id_x")) {
        return {"LocalInvocationId", 0};
    } else if (ends_with(name, ".__thread_id_y")) {
        return {"LocalInvocationId", 1};
    } else if (ends_with(name, ".__thread_id_z")) {
        return {"LocalInvocationId", 2};
    } else if (ends_with(name, ".__block_id_x")) {
        return {"WorkgroupId", 0};
    } else if (ends_with(name, ".__block_id_y")) {
        return {"WorkgroupId", 1};
    } else if (ends_with(name, ".__block_id_z")) {
        return {"WorkgroupId", 2};
    } else if (ends_with(name, "id_w")) {
        user_error << "Vulkan only supports <=3 dimensions for gpu blocks";
    }
    internal_error << "simt_intrinsic called on bad variable name: " << name << "\n";
    return {"", -1};
}
int thread_loop_workgroup_index(const std::string &name) {
    std::string ids[] = {".__thread_id_x",
                         ".__thread_id_y",
                         ".__thread_id_z"};
    for (size_t i = 0; i < sizeof(ids) / sizeof(std::string); i++) {
        if (ends_with(name, ids[i])) { return i; }
    }
    return -1;
}
}  // anonymous namespace

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const For *op) {

    if (is_gpu_var(op->name)) {
        internal_assert((op->for_type == ForType::GPUBlock) ||
                        (op->for_type == ForType::GPUThread))
            << "kernel loops must be either gpu block or gpu thread\n";
        // This should always be true at this point in codegen
        internal_assert(is_const_zero(op->min));

        // Save & validate the workgroup size
        int idx = thread_loop_workgroup_index(op->name);
        if (idx >= 0) {
            const IntImm *wsize = op->extent.as<IntImm>();
            user_assert(wsize != nullptr) << "Vulkan requires statically-known workgroup size.\n";
            uint32_t new_wsize = wsize->value;
            user_assert(workgroup_size[idx] == 0 || workgroup_size[idx] == new_wsize) << "Vulkan requires all kernels have the same workgroup size, but two different ones "
                                                                                         "were encountered "
                                                                                      << workgroup_size[idx] << " and " << new_wsize << " in dimension " << idx << "\n";
            workgroup_size[idx] = new_wsize;
        }

        auto intrinsic = simt_intrinsic(op->name);

        // Intrinsics are inserted when adding the kernel
        internal_assert(symbol_table.contains(intrinsic.first));

        uint32_t intrinsic_id = symbol_table.get(intrinsic.first).first;
        uint32_t gpu_var_id = next_id++;
        uint32_t unsigned_gpu_var_id = next_id++;
        add_instruction(SpvOpCompositeExtract, {map_type(UInt(32)), unsigned_gpu_var_id, intrinsic_id, intrinsic.second});
        // cast to int, which is what's expected by Halide's for loops
        add_instruction(SpvOpBitcast, {map_type(Int(32)), gpu_var_id, unsigned_gpu_var_id});

        {
            ScopedBinding<std::pair<uint32_t, uint32_t>> binding(symbol_table, op->name, {gpu_var_id, SpvStorageClassUniform});
            op->body.accept(this);
        }

    } else {

        internal_assert(op->for_type == ForType::Serial) << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit unhandled For type: " << op->for_type << "\n";

        // TODO: Loop vars are alway int32_t right?
        uint32_t index_type_id = map_type(Int(32));
        uint32_t index_var_type_id = map_pointer_type(Int(32), SpvStorageClassFunction);

        op->min.accept(this);
        uint32_t min_id = id;
        op->extent.accept(this);
        uint32_t extent_id = id;

        // Compute max.
        uint32_t max_id = next_id++;
        add_instruction(SpvOpIAdd, {index_type_id, max_id, min_id, extent_id});

        // Declare loop var
        // TODO: Can we use the phi node for this?
        uint32_t loop_var_id = next_id++;
        add_allocation(index_var_type_id, loop_var_id, SpvStorageClassFunction, min_id);

        uint32_t header_label_id = next_id++;
        uint32_t loop_top_label_id = next_id++;
        uint32_t body_label_id = next_id++;
        uint32_t continue_label_id = next_id++;
        uint32_t merge_label_id = next_id++;
        add_instruction(SpvOpLabel, {header_label_id});
        add_instruction(SpvOpLoopMerge, {merge_label_id, continue_label_id, SpvLoopControlMaskNone});
        add_instruction(SpvOpBranch, {loop_top_label_id});
        add_instruction(SpvOpLabel, {loop_top_label_id});

        // loop test.
        uint32_t cur_index_id = next_id++;
        add_instruction(SpvOpLoad, {index_type_id, cur_index_id, loop_var_id});

        uint32_t loop_test_id = next_id++;
        add_instruction(SpvOpSLessThanEqual, {loop_test_id, cur_index_id, max_id});
        add_instruction(SpvOpBranchConditional, {loop_test_id, body_label_id, merge_label_id});

        add_instruction(SpvOpLabel, {body_label_id});

        {
            ScopedBinding<std::pair<uint32_t, uint32_t>> binding(symbol_table, op->name, {cur_index_id, SpvStorageClassFunction});

            op->body.accept(this);
        }

        add_instruction(SpvOpBranch, {continue_label_id});
        add_instruction(SpvOpLabel, {continue_label_id});

        // Loop var update?
        uint32_t next_index_id = next_id++;
        int32_t one = 1;
        uint32_t constant_one_id = emit_constant(Int(32), &one);
        add_instruction(SpvOpIAdd, {index_type_id, next_index_id, cur_index_id, constant_one_id});
        add_instruction(SpvOpStore, {index_type_id, next_index_id, loop_var_id});
        add_instruction(SpvOpBranch, {header_label_id});
        add_instruction(SpvOpLabel, {merge_label_id});
    }
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Ramp *op) {
    // TODO: Is there a way to do this that doesn't require duplicating lane values?
    uint32_t base_type_id = map_type(op->base.type());
    uint32_t type_id = map_type(op->type);
    op->base.accept(this);
    uint32_t base_id = id;
    op->stride.accept(this);
    uint32_t stride_id = id;
    uint32_t add_opcode = op->base.type().is_float() ? SpvOpFAdd : SpvOpIAdd;
    // Generate adds to make the elements of the ramp.
    uint32_t prev_id = base_id;
    uint32_t first_id = next_id;
    for (int i = 1; i < op->lanes; i++) {
        uint32_t this_id = next_id++;
        add_instruction(add_opcode, {base_type_id, this_id, prev_id, stride_id});
        prev_id = this_id;
    }

    id = next_id++;
    spir_v_kernels.push_back(((op->lanes + 3) << 16) | SpvOpCompositeConstruct);
    spir_v_kernels.push_back(type_id);
    spir_v_kernels.push_back(id);
    spir_v_kernels.push_back(base_id);
    for (int i = 1; i < op->lanes; i++) {
        spir_v_kernels.push_back(first_id++);
    }
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Broadcast *op) {
    // TODO: Is there a way to do this that doesn't require duplicating lane values?
    uint32_t type_id = map_type(op->type);
    op->value.accept(this);
    uint32_t value_id = id;
    id = next_id++;
    spir_v_kernels.push_back(((op->lanes + 3) << 16) | SpvOpCompositeConstruct);
    spir_v_kernels.push_back(type_id);
    spir_v_kernels.push_back(id);
    spir_v_kernels.insert(spir_v_kernels.end(), op->lanes, value_id);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Provide *) {
    internal_error << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Provide *): Provide encountered during codegen\n";
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Allocate *) {
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Free *) {
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Realize *) {
    internal_error << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Realize *): Realize encountered during codegen\n";
}

template<typename StmtOrExpr>
CodeGen_Vulkan_Dev::SPIRVEmitter::PhiNodeInputs
CodeGen_Vulkan_Dev::SPIRVEmitter::emit_if_then_else(const Expr &condition,
                                                    StmtOrExpr then_case, StmtOrExpr else_case) {
    condition.accept(this);
    uint32_t cond_id = id;
    uint32_t then_label_id = next_id++;
    uint32_t else_label_id = next_id++;
    uint32_t merge_label_id = next_id++;

    add_instruction(SpvOpSelectionMerge, {merge_label_id, SpvSelectionControlMaskNone});
    add_instruction(SpvOpBranchConditional, {cond_id, then_label_id, else_label_id});
    add_instruction(SpvOpLabel, {then_label_id});

    then_case.accept(this);
    uint32_t then_id = id;

    add_instruction(SpvOpBranch, {merge_label_id});
    add_instruction(SpvOpLabel, {else_label_id});

    else_case.accept(this);
    uint32_t else_id = id;

    // Every basic block must end with a branch instruction
    add_instruction(SpvOpBranch, {merge_label_id});

    add_instruction(SpvOpLabel, {merge_label_id});

    return {{then_id, then_label_id, else_id, else_label_id}};
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const IfThenElse *op) {
    emit_if_then_else(op->condition, op->then_case, op->else_case);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Evaluate *op) {
    op->value.accept(this);
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Shuffle *op) {
    internal_assert(op->vectors.size() == 2) << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Shuffle *op): SPIR-V codegen currently only supports shuffles of vector pairs.\n";
    uint32_t type_id = map_type(op->type);
    op->vectors[0].accept(this);
    uint32_t vector0_id = id;
    op->vectors[1].accept(this);
    uint32_t vector1_id = id;

    id = next_id++;
    spir_v_kernels.push_back(((5 + op->indices.size()) << 16) | SpvOpPhi);
    spir_v_kernels.push_back(type_id);
    spir_v_kernels.push_back(id);
    spir_v_kernels.push_back(vector0_id);
    spir_v_kernels.push_back(vector1_id);
    spir_v_kernels.insert(spir_v_kernels.end(), op->indices.begin(), op->indices.end());
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Prefetch *) {
    internal_error << "CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Prefetch *): Prefetch encountered during codegen\n";
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Fork *) {
    internal_error << "void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Fork *) not supported yet.";
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Acquire *) {
    internal_error << "void CodeGen_Vulkan_Dev::SPIRVEmitter::visit(const Acquire *) not supported yet.";
}

// TODO: fast math decorations.
void CodeGen_Vulkan_Dev::SPIRVEmitter::visit_binop(Type t, const Expr &a, const Expr &b, uint32_t opcode) {
    uint32_t type_id = map_type(t);
    a.accept(this);
    uint32_t a_id = id;
    b.accept(this);
    uint32_t b_id = id;
    id = next_id++;
    add_instruction(opcode, {type_id, id, a_id, b_id});
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::add_allocation(uint32_t result_type_id,
                                                      uint32_t result_id,
                                                      uint32_t storage_class,
                                                      uint32_t initializer) {
    if (initializer) {
        add_instruction(spir_v_kernel_allocations, SpvOpVariable, {result_type_id, result_id, storage_class, initializer});
    } else {
        add_instruction(spir_v_kernel_allocations, SpvOpVariable, {result_type_id, result_id, storage_class});
    }
}

void CodeGen_Vulkan_Dev::SPIRVEmitter::add_kernel(const Stmt &s,
                                                  const std::string &name,
                                                  const std::vector<DeviceArgument> &args) {
    debug(2) << "Adding Vulkan kernel " << name << "\n";

    // Add function definition
    // TODO: can we use one of the function control annotations?

    // We'll discover the workgroup size as we traverse the kernel
    workgroup_size[0] = 0;
    workgroup_size[1] = 0;
    workgroup_size[2] = 0;

    // Declare the function type.  TODO: should this be unique?
    uint32_t function_type_id = next_id++;

    add_instruction(spir_v_types, SpvOpTypeFunction, {function_type_id, void_id});

    // Add definition and parameters
    current_function_id = next_id++;
    add_instruction(SpvOpFunction, {void_id, current_function_id, SpvFunctionControlMaskNone, function_type_id});

    // Insert the starting label
    add_instruction(SpvOpLabel, {next_id++});

    // TODO: what about variables that need the SIMT intrinsics for their initializer?
    // Save the location where we'll insert OpVariable instructions
    size_t index = spir_v_kernels.size();

    std::vector<uint32_t> entry_point_interface;
    entry_point_interface.push_back(SpvExecutionModelGLCompute);
    entry_point_interface.push_back(current_function_id);
    // Add the string name of the function
    encode_string(entry_point_interface, (name.size() + 1 + 3) / 4, name.size(), name.c_str());

    // TODO: only add the SIMT intrinsics used
    auto intrinsics = {"WorkgroupId", "LocalInvocationId"};
    for (const auto *intrinsic : intrinsics) {
        uint32_t intrinsic_id = next_id++;
        uint32_t intrinsic_loaded_id = next_id++;
        // The builtins are pointers to vec3
        uint32_t intrinsic_type_id = map_pointer_type(Type(Type::UInt, 32, 3), SpvStorageClassInput);

        add_instruction(spir_v_types, SpvOpVariable, {intrinsic_type_id, intrinsic_id, SpvStorageClassInput});
        add_instruction(SpvOpLoad, {map_type(Type(Type::UInt, 32, 3)), intrinsic_loaded_id, intrinsic_id});
        symbol_table.push(intrinsic, {intrinsic_loaded_id, SpvStorageClassInput});

        // Annotate that this is the specific builtin
        auto built_in_kind = starts_with(intrinsic, "Workgroup") ? SpvBuiltInWorkgroupId : SpvBuiltInLocalInvocationId;
        add_instruction(spir_v_annotations, SpvOpDecorate, {intrinsic_id, SpvDecorationBuiltIn, built_in_kind});

        // Add the builtin to the interface
        entry_point_interface.push_back(intrinsic_id);
    }

    // Add the entry point and exection mode
    add_instruction(spir_v_entrypoints,
                    SpvOpEntryPoint, entry_point_interface);

    // GLSL-style: each input buffer is a runtime array in a buffer struct
    // All other params get passed in as a single uniform block
    // First, need to count scalar parameters to construct the uniform struct
    std::vector<uint32_t> scalar_types;
    uint32_t offset = 0;
    uint32_t param_pack_type_id = next_id++;
    uint32_t param_pack_ptr_type_id = next_id++;
    uint32_t param_pack_id = next_id++;
    scalar_types.push_back(param_pack_type_id);
    for (const auto &arg : args) {
        if (!arg.is_buffer) {
            // record the type for later constructing the params struct type
            scalar_types.push_back(map_type(arg.type));

            // Add a decoration describing the offset
            add_instruction(spir_v_annotations, SpvOpMemberDecorate, {param_pack_type_id, (uint32_t)(scalar_types.size() - 2), SpvDecorationOffset, offset});
            offset += arg.type.bytes();
        }
    }

    // Add a Block decoration for the parameter pack itself
    add_instruction(spir_v_annotations, SpvOpDecorate, {param_pack_type_id, SpvDecorationBlock});
    // We always pass in the parameter pack as the first binding
    add_instruction(spir_v_annotations, SpvOpDecorate, {param_pack_id, SpvDecorationDescriptorSet, 0});
    add_instruction(spir_v_annotations, SpvOpDecorate, {param_pack_id, SpvDecorationBinding, 0});

    // Add a struct type for the parameter pack and a pointer to it
    add_instruction(spir_v_types, SpvOpTypeStruct, scalar_types);
    add_instruction(spir_v_types, SpvOpTypePointer, {param_pack_ptr_type_id, SpvStorageClassUniform, param_pack_type_id});
    // Add a variable for the parameter pack
    add_instruction(spir_v_types, SpvOpVariable, {param_pack_ptr_type_id, param_pack_id, SpvStorageClassUniform});

    uint32_t binding_counter = 1;
    uint32_t scalar_index = 0;
    for (const auto &arg : args) {
        uint32_t param_id = next_id++;
        if (arg.is_buffer) {
            uint32_t element_type = map_type(arg.type);
            uint32_t runtime_arr_type = next_id++;
            uint32_t struct_type = next_id++;
            uint32_t ptr_struct_type = next_id++;
            add_instruction(spir_v_types, SpvOpTypeRuntimeArray, {runtime_arr_type, element_type});
            add_instruction(spir_v_types, SpvOpTypeStruct, {struct_type, runtime_arr_type});
            add_instruction(spir_v_types, SpvOpTypePointer, {ptr_struct_type, SpvStorageClassUniform, struct_type});
            // Annotate the struct to indicate it's passed in a GLSL-style buffer block
            add_instruction(spir_v_annotations, SpvOpDecorate, {struct_type, SpvDecorationBufferBlock});
            // Annotate the array with its stride
            add_instruction(spir_v_annotations, SpvOpDecorate, {runtime_arr_type, SpvDecorationArrayStride, (uint32_t)(arg.type.bytes())});
            // Annotate the offset for the array
            add_instruction(spir_v_annotations, SpvOpMemberDecorate, {struct_type, 0, SpvDecorationOffset, (uint32_t)0});

            // Set DescriptorSet and Binding
            add_instruction(spir_v_annotations, SpvOpDecorate, {param_id, SpvDecorationDescriptorSet, 0});
            add_instruction(spir_v_annotations, SpvOpDecorate, {param_id, SpvDecorationBinding, binding_counter++});

            add_instruction(spir_v_types, SpvOpVariable, {ptr_struct_type, param_id, SpvStorageClassUniform});
        } else {
            uint32_t access_chain_id = next_id++;
            add_instruction(SpvOpInBoundsAccessChain, {map_pointer_type(arg.type, SpvStorageClassUniform),
                                                       access_chain_id,
                                                       param_pack_id,
                                                       emit_constant(UInt(32), &scalar_index)});
            scalar_index++;
            add_instruction(SpvOpLoad, {map_type(arg.type), param_id, access_chain_id});
        }
        symbol_table.push(arg.name, {param_id, SpvStorageClassUniform});
    }

    s.accept(this);

    // Insert return and  function end delimiter
    add_instruction(SpvOpReturn, {});
    add_instruction(SpvOpFunctionEnd, {});

    // Insert the allocations in the right place
    auto it = spir_v_kernels.begin() + index;
    spir_v_kernels.insert(it, spir_v_kernel_allocations.begin(), spir_v_kernel_allocations.end());
    spir_v_kernel_allocations.clear();

    workgroup_size[0] = std::max(workgroup_size[0], (uint32_t)1);
    workgroup_size[1] = std::max(workgroup_size[1], (uint32_t)1);
    workgroup_size[2] = std::max(workgroup_size[2], (uint32_t)1);
    // Add workgroup size to execution mode
    add_instruction(spir_v_execution_modes, SpvOpExecutionMode,
                    {current_function_id, SpvExecutionModeLocalSize,
                     workgroup_size[0], workgroup_size[1], workgroup_size[2]});

    // Pop scope
    for (const auto &arg : args) {
        symbol_table.pop(arg.name);
    }

    // Reset to an invalid value for safety.
    current_function_id = 0;
}

CodeGen_Vulkan_Dev::CodeGen_Vulkan_Dev(Target t) {
}

namespace {
void add_extension(const std::string &extension_name, std::vector<uint32_t> &section) {
    uint32_t extra_words = (extension_name.size() + 1 + 3) / 4;
    section.push_back(((1 + extra_words) << 16) | SpvOpExtension);

    const char *data_temp = (const char *)extension_name.c_str();
    const size_t data_size = extension_name.size();
    encode_string(section, extra_words, data_size, data_temp);
}
}  // namespace
void CodeGen_Vulkan_Dev::init_module() {
    debug(2) << "Vulkan device codegen init_module\n";

    // Header.
    emitter.spir_v_header.push_back(SpvMagicNumber);
    emitter.spir_v_header.push_back(SpvVersion);
    emitter.spir_v_header.push_back(SpvSourceLanguageUnknown);
    emitter.spir_v_header.push_back(0);  // Bound placeholder
    emitter.spir_v_header.push_back(0);  // Reserved for schema.

    // the unique void type
    emitter.next_id++;  // 0 is not a valid id
    emitter.void_id = emitter.next_id++;
    emitter.add_instruction(emitter.spir_v_types, SpvOpTypeVoid, {emitter.void_id});

    // Capabilities
    // TODO: only add those required by the generated code
    emitter.add_instruction(emitter.spir_v_header, SpvOpCapability, {SpvCapabilityShader});
    //emitter.add_instruction(emitter.spir_v_header, SpvOpCapability, {SpvCapabilityInt8});
    //emitter.add_instruction(emitter.spir_v_header, SpvOpCapability, {SpvCapabilityUniformAndStorageBuffer8BitAccess});

    // Extensions
    // TODO: only add those required by the generated code
    add_extension(std::string("SPV_KHR_8bit_storage"), emitter.spir_v_header);

    // Memory model
    // TODO: 32-bit or 64-bit?
    // TODO: Which memory model?
    emitter.add_instruction(emitter.spir_v_header, SpvOpMemoryModel,
                            {SpvAddressingModelLogical, SpvMemoryModelGLSL450});

    // OpCapability instructions
    //    Enumerate type maps and add subwidth integer types if used
    // OpExtensions instructions
    // OpExtImport instructions
    // One OpMemoryModelInstruction
    // OpEntryPoint instructions -- tricky as we don't know them until the kernels are added. May need to insert as we go.
    // OpExecutionMode or OpExecutionModeId -- are these also added at add_kernel time?
    // debug -- empty?
    // annotation
    //     I believe alignment info for load/store/etc. is done with annotations.
    //     Also need various annotations for SIMT intrinsics, struct layouts, etc
    // OpType instructions. Contained in spir_v_types member.
    // Function declarations. Are there any?
    // Function bodies -- one per add_kernel
}

void CodeGen_Vulkan_Dev::add_kernel(Stmt stmt,
                                    const std::string &name,
                                    const std::vector<DeviceArgument> &args) {
    current_kernel_name = name;
    emitter.add_kernel(stmt, name, args);
    //dump();
}

std::vector<char> CodeGen_Vulkan_Dev::compile_to_src() {
    //#ifdef WITH_VULKAN

    emitter.spir_v_header[3] = emitter.next_id;

    std::vector<char> final_module;
    size_t total_size = (emitter.spir_v_header.size() + emitter.spir_v_entrypoints.size() + emitter.spir_v_execution_modes.size() + emitter.spir_v_annotations.size() + emitter.spir_v_types.size() + emitter.spir_v_kernels.size()) * sizeof(uint32_t);
    final_module.reserve(total_size);
    final_module.insert(final_module.end(), (const char *)emitter.spir_v_header.data(), (const char *)(emitter.spir_v_header.data() + emitter.spir_v_header.size()));
    final_module.insert(final_module.end(), (const char *)emitter.spir_v_entrypoints.data(), (const char *)(emitter.spir_v_entrypoints.data() + emitter.spir_v_entrypoints.size()));
    final_module.insert(final_module.end(), (const char *)emitter.spir_v_execution_modes.data(), (const char *)(emitter.spir_v_execution_modes.data() + emitter.spir_v_execution_modes.size()));
    final_module.insert(final_module.end(), (const char *)emitter.spir_v_annotations.data(), (const char *)(emitter.spir_v_annotations.data() + emitter.spir_v_annotations.size()));
    final_module.insert(final_module.end(), (const char *)emitter.spir_v_types.data(), (const char *)(emitter.spir_v_types.data() + emitter.spir_v_types.size()));
    final_module.insert(final_module.end(), (const char *)emitter.spir_v_kernels.data(), (const char *)(emitter.spir_v_kernels.data() + emitter.spir_v_kernels.size()));
    assert(final_module.size() == total_size);
    std::ofstream f("out.spv", std::ios::out | std::ios::binary);
    f.write((char *)(final_module.data()), final_module.size());
    f.close();

    return final_module;

    //#endif
}

std::string CodeGen_Vulkan_Dev::get_current_kernel_name() {
    return current_kernel_name;
}

std::string CodeGen_Vulkan_Dev::print_gpu_name(const std::string &name) {
    return name;
}

void CodeGen_Vulkan_Dev::dump() {
    // TODO: Figure out what goes here.
    // For now: dump to file so source can be consumed by validator
    auto module = compile_to_src();
    std::ofstream f("out.spv", std::ios::out | std::ios::binary);
    f.write((char *)(module.data()), module.size());
    f.close();
}

}  // namespace

std::unique_ptr<CodeGen_GPU_Dev> new_CodeGen_Vulkan_Dev(const Target &target) {
    return std::make_unique<CodeGen_Vulkan_Dev>(target);
}

}  // namespace Internal
}  // namespace Halide
