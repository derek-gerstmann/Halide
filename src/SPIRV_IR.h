#ifndef HALIDE_SPIRV_IR_H
#define HALIDE_SPIRV_IR_H

#include <vector>
#include <unordered_map>

#include "IntrusivePtr.h"

#include "spirv/1.0/spirv.h"


namespace Halide {
namespace Internal {

class SpvModule;
class SpvFunction;
class SpvBlock;
class SpvInstruction;
class SpvContext;

struct SpvModuleContents;
struct SpvFunctionContents;
struct SpvBlockContents;
struct SpvInstructionContents;

using SpvModuleContentsPtr = IntrusivePtr<SpvModuleContents>;
using SpvFunctionContentsPtr = IntrusivePtr<SpvFunctionContents>;
using SpvBlockContentsPtr = IntrusivePtr<SpvBlockContents>;
using SpvInstructionContentsPtr = IntrusivePtr<SpvInstructionContents>;

using SpvId = uint32_t;
using SpvBinary = std::vector<uint32_t>;

static constexpr SpvId SpvNoResult = 0;
static constexpr SpvId SpvNoType = 0;

// --

class SpvInstruction {
public:

    SpvInstruction();
    SpvInstruction(SpvOp op_code, SpvId result_id = SpvNoResult, SpvId type_id = SpvNoType);
    ~SpvInstruction() = default;

    SpvInstruction(const SpvInstruction &) = default;
    SpvInstruction &operator=(const SpvInstruction &) = default;
    SpvInstruction(SpvInstruction &&) = default;
    SpvInstruction &operator=(SpvInstruction &&) = default;


    void set_block(SpvBlock block);
    void set_result_id(SpvId id);
    void set_op_code(SpvOp oc);
    void add_operand(SpvId id);
    void add_immediate(SpvId id);
    
    SpvId result_id() const;
    SpvOp op_code() const;
    SpvId operand(uint32_t index);

    bool has_type(void) const;
    bool has_result(void) const;
    bool is_immediate(uint32_t index) const;
    uint32_t length() const;
    SpvBlock block() const;
    
    void encode(SpvBinary& binary) const;
    
protected:
    SpvInstructionContentsPtr contents;
};

// --

class SpvBlock {
public:
    SpvBlock();
    SpvBlock(SpvFunction func, SpvId id);
    ~SpvBlock() = default;

    SpvBlock(const SpvBlock &) = default;
    SpvBlock &operator=(const SpvBlock &) = default;
    SpvBlock(SpvBlock &&) = default;
    SpvBlock &operator=(SpvBlock &&) = default;

    void add_instruction(SpvInstruction inst);
    void add_variable(SpvInstruction var);
    void set_function(SpvFunction func);
    SpvFunction function() const;
    bool is_reachable() const;
    bool is_terminated() const;

    SpvId id() const;

    void encode(SpvBinary& binary) const;

protected:
   
    SpvBlockContentsPtr contents;
};

// --

class SpvFunction {
public:
    SpvFunction();
    SpvFunction(SpvModule module, SpvId func_id, SpvId result_type_id, SpvId func_type_id);
    ~SpvFunction() = default;

    SpvFunction(const SpvFunction &) = default;
    SpvFunction &operator=(const SpvFunction &) = default;
    SpvFunction(SpvFunction &&) = default;
    SpvFunction &operator=(SpvFunction &&) = default;

    void add_parameter(SpvInstruction param_ptr);

    void set_module(SpvModule m);
    SpvInstruction instruction() const;
    SpvModule module() const;
    SpvId id() const;

    void encode(SpvBinary& binary) const;

protected:
    SpvFunctionContentsPtr contents;
};

// --

class SpvModule {
public:
    SpvModule();
    ~SpvModule() = default;

    SpvModule(const SpvModule &) = default;
    SpvModule &operator=(const SpvModule &) = default;
    SpvModule(SpvModule &&) = default;
    SpvModule &operator=(SpvModule &&) = default;

    SpvFunction add_function(SpvId func_id, SpvId result_type, SpvId func_type, SpvId first_param);
    SpvId map_instruction(SpvInstruction inst);
    SpvInstruction lookup_instruction(SpvId result_id) const;

    void encode(SpvBinary& binary) const;

protected:
    SpvModuleContentsPtr contents;
};

// --

struct SpvInstructionContents {
    using Operands = std::vector<SpvId>;
    using Immediates = std::vector<bool>;
    mutable RefCount ref_count;
    SpvOp op_code = SpvOpNop;
    SpvId result_id = SpvNoResult;
    SpvId type_id = SpvNoType;
    Operands operands;
    Immediates immediates;
    SpvBlock block;
};

struct SpvBlockContents {
    using Instructions = std::vector<SpvInstruction>;
    using Variables = std::vector<SpvInstruction>;
    using Blocks = std::vector<SpvBlock>;
    mutable RefCount ref_count;
    SpvFunction parent;
    Instructions instructions;
    Variables variables;
    Blocks before;
    Blocks after;
    bool reachable = true;    
};

struct SpvFunctionContents {
    using Parameters = std::vector<SpvInstruction>;
    using Blocks = std::vector<SpvBlock>;
    mutable RefCount ref_count;
    SpvModule parent;
    SpvInstruction instruction;
    Parameters parameters;
    Blocks blocks; 
};

struct SpvModuleContents {
    using Functions = std::vector<SpvFunction>;
    using Instructions = std::vector<SpvInstruction>;
    using InstructionMap = std::unordered_map<SpvId, SpvInstruction>;
    mutable RefCount ref_count;
    Functions functions;
    Instructions instructions;
    InstructionMap instruction_map;
};

SpvInstruction::SpvInstruction() : contents(new SpvInstructionContents()) {

}

SpvInstruction::SpvInstruction(SpvOp op_code, SpvId result_id, SpvId type_id) 
    : contents(new SpvInstructionContents()) {
    contents->op_code = op_code;
    contents->result_id = result_id;
    contents->type_id = type_id;
}  

void SpvInstruction::set_block(SpvBlock block) { 
    contents->block = block;
}

void SpvInstruction::set_result_id(SpvId result_id) { 
    contents->result_id = result_id; 
}

void SpvInstruction::set_op_code(SpvOp op_code) { 
    contents->op_code = op_code; 
}

void SpvInstruction::add_operand(SpvId id) {
    contents->operands.push_back(id);
    contents->immediates.push_back(false);
}

void SpvInstruction::add_immediate(SpvId id) {
    contents->operands.push_back(id);
    contents->immediates.push_back(true);
}

SpvId SpvInstruction::result_id() const { 
    return contents->result_id; 
}

SpvOp SpvInstruction::op_code() const { 
    return contents->op_code; 
}

SpvId SpvInstruction::operand(uint32_t index) { 
    return contents->operands[index]; 
}

bool SpvInstruction::has_type(void) const { 
    return contents->type_id != SpvNoType; 
}

bool SpvInstruction::has_result(void) const { 
    return contents->result_id != SpvNoResult; 
}

bool SpvInstruction::is_immediate(uint32_t index) const { 
    return contents->immediates[index]; 
}

uint32_t SpvInstruction::length() const { 
    return (uint32_t)contents->operands.size(); 
}

SpvBlock SpvInstruction::block() const { 
    return contents->block; 
}

void SpvInstruction::encode(SpvBinary& binary) const {

    // Count the number of 32-bit words to represent the instruction
    uint32_t word_count = 1; 
    word_count += has_type() ? 1 : 0;
    word_count += has_result() ? 1 : 0;
    word_count += length();

    // Preface the instruction with the format
    // - high 16-bits indicate instruction length (number of 32-bit words)
    // - low 16-bits indicate op code
    binary.push_back(((word_count) << SpvWordCountShift) | contents->op_code);
    if(has_type()) { binary.push_back(contents->type_id); }
    if(has_result()) { binary.push_back(contents->result_id); }
    for(SpvId id : contents->operands) { binary.push_back(id); }
}

// --

template<>
RefCount &ref_count<SpvInstructionContents>(const SpvInstructionContents *c) noexcept {
    return c->ref_count;
}

template<>
void destroy<SpvInstructionContents>(const SpvInstructionContents *c) {
    delete c;
}

// --

SpvBlock::SpvBlock() 
    : contents( new SpvBlockContents() ) {

}

SpvBlock::SpvBlock(SpvFunction func, SpvId id) 
    : contents( new SpvBlockContents() ) {
    
    contents->parent = func;
    SpvInstruction inst(SpvOpLabel, id); // add a label for this block
    add_instruction(inst);
}

void SpvBlock::add_instruction(SpvInstruction inst) {
    inst.set_block(*this);
    contents->instructions.push_back(inst);        
}

void SpvBlock::add_variable(SpvInstruction var) {
    var.set_block(*this);
    contents->instructions.push_back(var);        
}

void SpvBlock::set_function(SpvFunction func) { 
    contents->parent = func; 
}

SpvFunction SpvBlock::function() const { 
    return contents->parent; 
}

bool SpvBlock::is_reachable() const { 
    return contents->reachable; 
}

bool SpvBlock::is_terminated() const {
    switch(contents->instructions.back().op_code()) {
        case SpvOpBranch:
        case SpvOpBranchConditional:
        case SpvOpSwitch:
        case SpvOpKill:
        case SpvOpReturn:
        case SpvOpReturnValue:
        case SpvOpUnreachable:
            return true;
        default:
            return false;
    };
}

SpvId SpvBlock::id() const { 
    return contents->instructions.front().result_id(); 
}

void SpvBlock::encode(SpvBinary& binary) const {
    contents->instructions.front().encode(binary);
    for(const SpvInstruction& var : contents->variables) {
        var.encode(binary);
    }
    for(size_t i = 1; i < contents->instructions.size(); ++i) { // skip label
        contents->instructions[i].encode(binary);
    }
}

// --

template<>
RefCount &ref_count<SpvBlockContents>(const SpvBlockContents *c) noexcept {
    return c->ref_count;
}

template<>
void destroy<SpvBlockContents>(const SpvBlockContents *c) {
    delete c;
}

// --

SpvFunction::SpvFunction() 
    : contents( new SpvFunctionContents() ) { }

SpvFunction::SpvFunction(SpvModule module, SpvId func_id, SpvId result_type_id, SpvId func_type_id)
    : contents( new SpvFunctionContents() ) {

    contents->parent = module;    
    contents->instruction = SpvInstruction(SpvOpFunction, func_id, result_type_id);
    contents->instruction.add_immediate(SpvFunctionControlMaskNone);
    contents->instruction.add_operand(func_type_id);
}

void SpvFunction::add_parameter(SpvInstruction param) {
    contents->parameters.push_back(param);        
}

void SpvFunction::set_module(SpvModule module) { 
    contents->parent = module; 
}

SpvInstruction SpvFunction::instruction() const {
    return contents->instruction;
}

SpvModule SpvFunction::module() const { 
    return contents->parent; 
}

SpvId SpvFunction::id() const { 
    return contents->instruction.result_id(); 
}

void SpvFunction::encode(SpvBinary& binary) const {
    contents->instruction.encode(binary);
    for(const SpvInstruction& param : contents->parameters) {
        param.encode(binary);
    }        
}

// --

template<>
RefCount &ref_count<SpvFunctionContents>(const SpvFunctionContents *c) noexcept {
    return c->ref_count;
}

template<>
void destroy<SpvFunctionContents>(const SpvFunctionContents *c) {
    delete c;
}

// --

SpvModule::SpvModule() 
    : contents( new SpvModuleContents() ) {

}

SpvFunction SpvModule::add_function(SpvId func_id, SpvId result_type_id, SpvId func_type_id, SpvId first_param) {
    SpvFunction func(*this, func_id, result_type_id, func_type_id);
    contents->functions.emplace_back(func);        
    map_instruction(func.instruction());

    SpvInstruction inst = lookup_instruction(func_type_id);
    if((inst.op_code() != SpvOpNop) && inst.length()) {
        uint32_t param_count = inst.length() - 1;
        for(uint32_t p = 0; p < param_count; ++p) {
            SpvInstruction param(
                SpvOpFunctionParameter, 
                first_param + p, 
                inst.operand(p + 1)
            );

            func.add_parameter(param);
            map_instruction(param);
        }
    }
    return func;
}

SpvId SpvModule::map_instruction(SpvInstruction inst) {
    const SpvId key = inst.result_id();
    if(contents->instruction_map.find(key) == contents->instruction_map.end()) {
        contents->instruction_map.insert({key, inst});
    } else {
        contents->instruction_map[key] = inst;
    }
    return key;
}

SpvInstruction SpvModule::lookup_instruction(SpvId result_id) const {
    auto it = contents->instruction_map.find(result_id);
    if(it == contents->instruction_map.end()) {
        return SpvInstruction();
    }
    return it->second;
}

void SpvModule::encode(SpvBinary& binary) const {
    for(const SpvFunction& func : contents->functions) {
        func.encode(binary);
    }
}

// --

template<>
RefCount &ref_count<SpvModuleContents>(const SpvModuleContents *c) noexcept {
    return c->ref_count;
}

template<>
void destroy<SpvModuleContents>(const SpvModuleContents *c) {
    delete c;
}

// --

class SpvContext {
public:

    SpvContext(SpvSourceLanguage src_lang, SpvAddressingModel addr_model, SpvMemoryModel mem_model);
    ~SpvContext() = default;

    SpvInstruction create_instruction();

    SpvId create_id();
    SpvId create_void_type();
    SpvId create_bool_type();
    SpvId create_pointer_type(SpvStorageClass storage_class, SpvId pointee);
    SpvId create_forward_pointer_type(SpvStorageClass storage_class, SpvId pointee);
    SpvId create_integer_type(int width, bool has_sign=true);
    SpvId create_float_type(int width);
    SpvId create_vector_type(SpvId component, int lanes);
    SpvId create_matrix_type(SpvId component, int rows, int cols);
    SpvId create_array_type(SpvId element, SpvId size, int stride);
    SpvId create_string_type(const char* str);

    void require(SpvCapability);
    bool is_required(SpvCapability) const;

protected:
    using StringMap = std::unordered_map<std::string, SpvId>;
    using Capabilities = std::set<SpvCapability>;
    using Functions = std::vector<SpvFunction>;
    using Instructions = std::vector<SpvInstruction>;

    SpvId id_counter = 0;
    SpvSourceLanguage source_language = SpvSourceLanguageUnknown;
    SpvAddressingModel addressing_model = SpvAddressingModelLogical;
    SpvMemoryModel memory_model = SpvMemoryModelSimple;

    SpvModule module;
    SpvFunction entry_point;    

    Capabilities requirements;
    Instructions globals;
    Instructions strings;
    Instructions names;
    Functions functions;

    StringMap string_map;
};

SpvContext::SpvContext(SpvSourceLanguage src_lang, SpvAddressingModel addr_model, SpvMemoryModel mem_model)
    : source_language(src_lang), addressing_model(addr_model), memory_model(mem_model) {

}
    
SpvId SpvContext::create_id() {
    return ++id_counter;
}

void SpvContext::require(SpvCapability capability) {
    if(requirements.find(capability) == requirements.end()) {
        requirements.insert(capability);
    }
}

bool SpvContext::is_required(SpvCapability capability) const {
    if(requirements.find(capability) != requirements.end()) {
        return true;
    }
    return false;
}

// --

}} // namespace: Halide::Internal

#endif // HALIDE_SPIRV_IR_H