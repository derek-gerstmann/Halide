#ifndef HALIDE_SPIRV_IR_H
#define HALIDE_SPIRV_IR_H

#include <vector>
#include <unordered_map>

#include "Type.h"
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

static constexpr SpvId SpvInvalidId = SpvId(-1);
static constexpr SpvId SpvNoResult = 0;
static constexpr SpvId SpvNoType = 0;

// --

class SpvInstruction {
public:

    SpvInstruction() = default;
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
    bool is_defined(void) const;
    bool is_immediate(uint32_t index) const;
    uint32_t length() const;
    SpvBlock block() const;
    
    void encode(SpvBinary& binary) const;
    
    static SpvInstruction make(SpvOp op_code, SpvId result_id = SpvNoResult, SpvId type_id = SpvNoType);
    static SpvInstruction make(SpvOp op_code, std::initializer_list<uint32_t> data);    

protected:
    SpvInstructionContentsPtr contents;
};

// --

class SpvBlock {
public:
    SpvBlock() = default;
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
    bool is_defined() const;
    SpvId id() const;

    void encode(SpvBinary& binary) const;

    static SpvBlock make(SpvFunction func, SpvId id);

protected:
   
    SpvBlockContentsPtr contents;
};

// --

class SpvFunction {
public:
    SpvFunction() = default;
    ~SpvFunction() = default;

    SpvFunction(const SpvFunction &) = default;
    SpvFunction &operator=(const SpvFunction &) = default;
    SpvFunction(SpvFunction &&) = default;
    SpvFunction &operator=(SpvFunction &&) = default;

    void add_parameter(SpvInstruction param_ptr);
    void set_module(SpvModule m);
    bool is_defined() const;
    SpvInstruction instruction() const;
    SpvModule module() const;
    SpvId id() const;

    void encode(SpvBinary& binary) const;

    static SpvFunction make(SpvModule module, SpvId func_id, SpvId result_type_id, SpvId func_type_id);
    static SpvFunction make(SpvId func_id, SpvId result_type_id, SpvId func_type_id);

protected:
    SpvFunctionContentsPtr contents;
};

// --

class SpvModule {
public:
    SpvModule() = default;
    ~SpvModule() = default;

    SpvModule(const SpvModule &) = default;
    SpvModule &operator=(const SpvModule &) = default;
    SpvModule(SpvModule &&) = default;
    SpvModule &operator=(SpvModule &&) = default;

    void add_debug(SpvInstruction val);    
    void add_annotation(SpvInstruction val);    
    void add_type(SpvInstruction val);
    void add_constant(SpvInstruction val);
    void add_global(SpvInstruction val);
    void add_function(SpvFunction val);
    void add_instruction(SpvInstruction val);

    void require_capability(SpvCapability val);
    void require_extension(const std::string& val);    

    void set_source_language(SpvSourceLanguage val);
    void set_addressing_model(SpvAddressingModel val);
    void set_memory_model(SpvMemoryModel val);
    void set_entry_point(SpvFunction val);
    void set_execution_mode(SpvInstruction val);

    SpvSourceLanguage source_language() const;
    SpvAddressingModel addressing_model() const;
    SpvMemoryModel memory_model() const;
    SpvFunction entry_point() const;    
    SpvInstruction execution_mode() const;
    SpvModule module() const;
    
    bool is_capability_required(SpvCapability val) const;
    bool is_extension_required(const std::string& val) const;
    bool is_defined() const;
    SpvId id() const;

    void encode(SpvBinary& binary) const;

    static SpvModule make(SpvId module_id, 
        SpvSourceLanguage source_language = SpvSourceLanguageUnknown,
        SpvAddressingModel addressing_model = SpvAddressingModelLogical,
        SpvMemoryModel memory_model = SpvMemoryModelSimple);

protected:
    SpvModuleContentsPtr contents;
};

// --

class SpvContext {
public:

    SpvContext();
    ~SpvContext() = default;

    SpvContext(const SpvContext &) = delete;
    SpvContext &operator=(const SpvContext &) = delete;

    SpvInstruction add_instruction(SpvInstruction val);
    SpvFunction add_function(SpvId func_id, SpvId result_type_id, SpvId func_type_id, SpvId first_param);

    SpvId make_id();
    SpvId make_void_type();
    SpvId make_bool_type();
    SpvId make_basic_type(const Type& type);
    SpvId make_pointer_type(SpvStorageClass storage_class, SpvId pointee);
    SpvId make_forward_pointer_type(SpvStorageClass storage_class, SpvId pointee);
    SpvId make_integer_type(int width, bool has_sign=true);
    SpvId make_float_type(int width);
    SpvId make_vector_type(SpvId component, int lanes);
    SpvId make_matrix_type(SpvId component, int rows, int cols);
    SpvId make_array_type(SpvId element, SpvId size, int stride);
    SpvId make_string_type(const char* str);

    void set_source_language(SpvSourceLanguage val);
    void set_addressing_model(SpvAddressingModel val);
    void set_memory_model(SpvMemoryModel val);

    SpvSourceLanguage source_language() const;
    SpvAddressingModel addressing_model() const;
    SpvMemoryModel memory_model() const;
    
    void require_extension(const std::string& extension);
    void require_capability(SpvCapability);

    bool is_extension_required(const std::string& extension) const;
    bool is_capability_required(SpvCapability) const;

    operator SpvModule() const;

protected:
    SpvId add_basic_type(const Type& type);
    SpvId lookup_basic_type(const Type& type) const;
    SpvId convert_basic_type(const Type& type) const;
    bool has_type(const Type& type) const;

    SpvId map_instruction(SpvInstruction inst);
    SpvInstruction lookup_instruction(SpvId result_id) const;
    bool has_instruction(SpvId inst) const;

    using TypeMap = std::map<Type, SpvId>;
    using StringMap = std::unordered_map<std::string, SpvId>;
    using InstructionMap = std::unordered_map<SpvId, SpvInstruction>;

    SpvId uid = 0;
    SpvModule module;
    TypeMap type_map;
    StringMap string_map;
    InstructionMap constant_map;
    InstructionMap instruction_map;
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
    using Capabilities = std::set<SpvCapability>;
    using Extensions = std::set<std::string>;
    using Imports = std::set<std::string>;
    using Functions = std::vector<SpvFunction>;
    using Instructions = std::vector<SpvInstruction>;

    mutable RefCount ref_count;
    SpvId module_id = SpvInvalidId;
    SpvSourceLanguage source_language = SpvSourceLanguageUnknown;
    SpvAddressingModel addressing_model = SpvAddressingModelLogical;
    SpvMemoryModel memory_model = SpvMemoryModelSimple;
    Capabilities capabilities;
    Extensions extensions;
    Imports imports;
    SpvFunction entry_point;    
    SpvInstruction execution_mode;
    Instructions debug;
    Instructions annotations;
    Instructions types;
    Instructions constants;
    Instructions globals;
    Functions functions;
    Instructions instructions;
};

SpvInstruction SpvInstruction::make(SpvOp op_code, SpvId result_id, SpvId type_id) {
    SpvInstruction instance;
    instance.contents = SpvInstructionContentsPtr(new SpvInstructionContents);
    instance.contents->op_code = op_code;
    instance.contents->result_id = result_id;
    instance.contents->type_id = type_id;
    return instance;
}  

SpvInstruction SpvInstruction::make(SpvOp op_code, std::initializer_list<uint32_t> data) {
    SpvInstruction instance;
    instance.contents = SpvInstructionContentsPtr(new SpvInstructionContents);
    instance.contents->op_code = op_code;
    instance.contents->operands.insert( instance.contents->operands.end(), data.begin(), data.end() );
    return instance;
}

void SpvInstruction::set_block(SpvBlock block) { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    contents->block = block;
}

void SpvInstruction::set_result_id(SpvId result_id) { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    contents->result_id = result_id; 
}

void SpvInstruction::set_op_code(SpvOp op_code) { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    contents->op_code = op_code; 
}

void SpvInstruction::add_operand(SpvId id) {
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    contents->operands.push_back(id);
    contents->immediates.push_back(false);
}

void SpvInstruction::add_immediate(SpvId id) {
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    contents->operands.push_back(id);
    contents->immediates.push_back(true);
}

SpvId SpvInstruction::result_id() const { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before accessing its properties\n";
    return contents->result_id; 
}

SpvOp SpvInstruction::op_code() const { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before accessing its properties\n";
    return contents->op_code; 
}

SpvId SpvInstruction::operand(uint32_t index) { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before accessing its properties\n";
    return contents->operands[index]; 
}

bool SpvInstruction::has_type(void) const { 
    if(!is_defined()) { return false; }
    return contents->type_id != SpvNoType; 
}

bool SpvInstruction::has_result(void) const { 
    if(!is_defined()) { return false; }
    return contents->result_id != SpvNoResult; 
}

bool SpvInstruction::is_defined() const {
    return contents.defined();
}

bool SpvInstruction::is_immediate(uint32_t index) const { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    return contents->immediates[index]; 
}

uint32_t SpvInstruction::length() const { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    return (uint32_t)contents->operands.size(); 
}

SpvBlock SpvInstruction::block() const { 
    user_assert(is_defined()) << "An SpvInstruction must be defined before modifying its properties\n";
    return contents->block; 
}

void SpvInstruction::encode(SpvBinary& binary) const {
    user_assert(is_defined()) << "An SpvInstruction must be defined before accessing its properties\n";

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

SpvBlock SpvBlock::make(SpvFunction func, SpvId block_id) {
    SpvBlock instance;
    instance.contents = SpvBlockContentsPtr( new SpvBlockContents() );
    instance.contents->parent = func;
    SpvInstruction inst = SpvInstruction::make(SpvOpLabel, block_id); // add a label for this block
    instance.add_instruction(inst);
    return instance;
}

void SpvBlock::add_instruction(SpvInstruction inst) {
    user_assert(is_defined()) << "An SpvBlock must be defined before modifying its properties\n";
    inst.set_block(*this);
    contents->instructions.push_back(inst);        
}

void SpvBlock::add_variable(SpvInstruction var) {
    user_assert(is_defined()) << "An SpvBlock must be defined before modifying its properties\n";
    var.set_block(*this);
    contents->instructions.push_back(var);        
}

void SpvBlock::set_function(SpvFunction func) { 
    user_assert(is_defined()) << "An SpvBlock must be defined before modifying its properties\n";
    contents->parent = func; 
}

SpvFunction SpvBlock::function() const { 
    user_assert(is_defined()) << "An SpvBlock must be defined before accessing its properties\n";
    return contents->parent; 
}

bool SpvBlock::is_reachable() const { 
    user_assert(is_defined()) << "An SpvBlock must be defined before accessing its properties\n";
    return contents->reachable; 
}

bool SpvBlock::is_defined() const { 
    return contents.defined();
}

bool SpvBlock::is_terminated() const {
    user_assert(is_defined()) << "An SpvBlock must be defined before accessing its properties\n";
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
    user_assert(is_defined()) << "An SpvBlock must be defined before accessing its properties\n";
    return contents->instructions.front().result_id(); 
}

void SpvBlock::encode(SpvBinary& binary) const {
    user_assert(is_defined()) << "An SpvBlock must be defined before accessing its properties\n";
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

SpvFunction SpvFunction::make(SpvModule module, SpvId func_id, SpvId result_type_id, SpvId func_type_id) {
    SpvFunction instance;
    instance.contents = SpvFunctionContentsPtr( new SpvFunctionContents() );
    instance.contents->parent = module;    
    instance.contents->instruction = SpvInstruction::make(SpvOpFunction, func_id, result_type_id);
    instance.contents->instruction.add_immediate(SpvFunctionControlMaskNone);
    instance.contents->instruction.add_operand(func_type_id);
    return instance;
}

bool SpvFunction::is_defined() const { 
    return contents.defined();
}

void SpvFunction::add_parameter(SpvInstruction param) {
    user_assert(is_defined()) << "An SpvFunction must be defined before accessing its properties\n";
    contents->parameters.push_back(param);        
}

void SpvFunction::set_module(SpvModule module) { 
    user_assert(is_defined()) << "An SpvFunction must be defined before accessing its properties\n";
    contents->parent = module; 
}

SpvInstruction SpvFunction::instruction() const {
    user_assert(is_defined()) << "An SpvFunction must be defined before accessing its properties\n";
    return contents->instruction;
}

SpvModule SpvFunction::module() const { 
    user_assert(is_defined()) << "An SpvFunction must be defined before accessing its properties\n";
    return contents->parent; 
}

SpvId SpvFunction::id() const { 
    user_assert(is_defined()) << "An SpvFunction must be defined before accessing its properties\n";
    return contents->instruction.result_id(); 
}

void SpvFunction::encode(SpvBinary& binary) const {
    user_assert(is_defined()) << "An SpvFunction must be defined before accessing its properties\n";
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

SpvModule SpvModule::make(SpvId module_id, 
    SpvSourceLanguage source_language,
    SpvAddressingModel addressing_model,
    SpvMemoryModel memory_model) {
    SpvModule instance;
    instance.contents = SpvModuleContentsPtr( new SpvModuleContents() );
    instance.contents->module_id = module_id;
    instance.contents->source_language = source_language;
    instance.contents->addressing_model = addressing_model;
    instance.contents->memory_model = memory_model;
    return instance;
}

bool SpvModule::is_defined() const { 
    return contents.defined();
}

void SpvModule::add_debug(SpvInstruction val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->debug.push_back(val);
}   

void SpvModule::add_annotation(SpvInstruction val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->annotations.push_back(val);
}   

void SpvModule::add_type(SpvInstruction val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->types.push_back(val);
}

void SpvModule::add_constant(SpvInstruction val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->constants.push_back(val);
}

void SpvModule::add_global(SpvInstruction val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->globals.push_back(val);
}

void SpvModule::add_instruction(SpvInstruction val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->instructions.push_back(val);
}

void SpvModule::add_function(SpvFunction val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->functions.push_back(val);
}

void SpvModule::set_source_language(SpvSourceLanguage val){
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->source_language = val;
}

void SpvModule::set_addressing_model(SpvAddressingModel val){
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->addressing_model = val;
}

void SpvModule::set_memory_model(SpvMemoryModel val) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    contents->memory_model = val;
}

SpvSourceLanguage SpvModule::source_language() const{
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    return contents->source_language;
}

SpvAddressingModel SpvModule::addressing_model() const {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    return contents->addressing_model;
}

SpvMemoryModel SpvModule::memory_model() const {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    return contents->memory_model;
}

void SpvModule::require_extension(const std::string& extension) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    if(contents->extensions.find(extension) == contents->extensions.end()) {
        contents->extensions.insert(extension);
    }
}

bool SpvModule::is_extension_required(const std::string& extension) const {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    if(contents->extensions.find(extension) != contents->extensions.end()) {
        return true;
    }
    return false;
}

void SpvModule::require_capability(SpvCapability capability) {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    if(contents->capabilities.find(capability) == contents->capabilities.end()) {
        contents->capabilities.insert(capability);
    }
}

bool SpvModule::is_capability_required(SpvCapability capability) const {
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    if(contents->capabilities.find(capability) != contents->capabilities.end()) {
        return true;
    }
    return false;
}

SpvId SpvModule::id() const { 
    user_assert(is_defined()) << "An SpvModule must be defined before accessing its properties\n";
    return contents->module_id; 
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

SpvContext::SpvContext() {
    module = SpvModule::make(++uid);
}
    
SpvId SpvContext::make_id() {
    return ++uid;
}

SpvId SpvContext::make_void_type() {
    Type type;
    SpvId result = lookup_basic_type(type);
    if(result == SpvInvalidId) {
        SpvId type_id = add_basic_type(type);
        SpvInstruction inst = SpvInstruction::make(SpvOpTypeVoid, type_id);
        module.add_type(inst);
        result = type_id;
    }
    return result;
}

SpvId SpvContext::make_bool_type() {
    Type type = Bool();
    SpvId result = lookup_basic_type(type);
    if(result == SpvInvalidId) {
        SpvId type_id = add_basic_type(type);
        SpvInstruction inst = SpvInstruction::make(SpvOpTypeBool, type_id);
        module.add_type(inst);
        result = type_id;
    }
    return result;
}

SpvId SpvContext::make_pointer_type(SpvStorageClass storage_class, SpvId pointee) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvId SpvContext::make_forward_pointer_type(SpvStorageClass storage_class, SpvId pointee) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvId SpvContext::make_integer_type(int width, bool has_sign) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvId SpvContext::make_float_type(int width) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvId SpvContext::make_vector_type(SpvId component, int lanes) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvId SpvContext::make_matrix_type(SpvId component, int rows, int cols) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvId SpvContext::make_array_type(SpvId element, SpvId size, int stride) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvId SpvContext::make_string_type(const char* str) {
    SpvId result = SpvInvalidId;
    return result;
}

SpvFunction SpvContext::add_function(SpvId func_id, SpvId result_type_id, SpvId func_type_id, SpvId first_param) {
    SpvFunction func = SpvFunction::make(module, func_id, result_type_id, func_type_id);
    module.add_function(func);        
    map_instruction(func.instruction());

    SpvInstruction inst = lookup_instruction(func_type_id);
    if((inst.op_code() != SpvOpNop) && inst.length()) {
        uint32_t param_count = inst.length() - 1;
        for(uint32_t p = 0; p < param_count; ++p) {
            SpvInstruction param = SpvInstruction::make(
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

SpvContext::operator SpvModule() const {
    return this->module;
}

void SpvContext::require_capability(SpvCapability capability) {
    if(!module.is_capability_required(capability)) {
        module.require_capability(capability);
    }
}

bool SpvContext::is_capability_required(SpvCapability capability) const {
    return module.is_capability_required(capability);
}

void SpvContext::require_extension(const std::string& extension) {
    if(!module.is_extension_required(extension)) {
        module.require_extension(extension);
    }
}

bool SpvContext::is_extension_required(const std::string& extension) const {
    return module.is_extension_required(extension);
}

SpvId SpvContext::map_instruction(SpvInstruction inst) {
    const SpvId key = inst.result_id();
    if(instruction_map.find(key) == instruction_map.end()) {
        instruction_map.insert({key, inst});
    } else {
        instruction_map[key] = inst;
    }
    return key;
}

SpvInstruction SpvContext::lookup_instruction(SpvId result_id) const {
    InstructionMap::const_iterator it = instruction_map.find(result_id);
    if(it == instruction_map.end()) {
        return SpvInstruction();
    }
    return it->second;
}

SpvId SpvContext::add_basic_type(const Type& type) {
    SpvId result = lookup_basic_type(type);
    if(result == SpvInvalidId) {
        SpvId type_id = make_id();
        type_map.insert({type, type_id});
        result = type_id;
    }
    return result;
}

SpvId SpvContext::lookup_basic_type(const Type& type) const {
    TypeMap::const_iterator it = type_map.find(type);
    if(it == type_map.end()) {
        return SpvInvalidId;
    }
    return it->second;
}

bool SpvContext::has_type(const Type& type) const {
    TypeMap::const_iterator it = type_map.find(type);
    if(it == type_map.end()) {
        return false;
    }
    return true;
}

SpvId SpvContext::convert_basic_type(const Type& type) const {
    SpvId result = SpvInvalidId;
    switch(type.code()) {
        case Type::Int:
            if(type.lanes() > 1) {
                result = SpvOpTypeVector;
            } else {
                result = SpvOpTypeInt;
            }
            break;
        case Type::UInt:
            if(type.lanes() > 1) {
                result = SpvOpTypeVector;
            } else {
                result = SpvOpTypeInt;
            }
            break;
        case Type::Float:
            if(type.lanes() > 1) {
                result = SpvOpTypeVector;
            } else {
                result = SpvOpTypeFloat;
            }
            break;
        case Type::BFloat:
            if(type.lanes() > 1) {
                result = SpvOpTypeVector;
            } else {
                result = SpvOpTypeFloat;
            }
            break;
        case Type::Handle:
            break;
        default:
            break;
    };
    return result;
}

// --

}} // namespace: Halide::Internal

#endif // HALIDE_SPIRV_IR_H