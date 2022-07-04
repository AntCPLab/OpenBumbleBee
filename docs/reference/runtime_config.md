# SPU Runtime Configuration

## Table of Contents



- Messages
    - [ExecutableProto](#executableproto)
    - [IrProto](#irproto)
    - [RuntimeConfig](#runtimeconfig)
    - [ShapeProto](#shapeproto)
    - [ValueProto](#valueproto)
    - [XlaMeta](#xlameta)
  


- Enums
    - [DataType](#datatype)
    - [FieldType](#fieldtype)
    - [IrType](#irtype)
    - [ProtocolKind](#protocolkind)
    - [PtType](#pttype)
    - [RuntimeConfig.ExpMode](#runtimeconfigexpmode)
    - [RuntimeConfig.LogMode](#runtimeconfiglogmode)
    - [RuntimeConfig.SigmoidMode](#runtimeconfigsigmoidmode)
    - [Visibility](#visibility)
  


- [Scalar Value Types](#scalar-value-types)



 <!-- end services -->

## Messages


### ExecutableProto
The executable format accepted by SPU runtime.

- Inputs should be prepared before running executable.
- Output is maintained after execution, and can be fetched by output name.

```python
  rt = spu.Runtime(...)            # create an spu runtime.
  rt.set_var('x', ...)             # set variable to the runtime.
  exe = spu.ExecutableProto(       # prepare the executable.
          name = 'balabala',
          input_names = ['x'],
          output_names = ['y'],
          code = ...)
  rt.run(exe)                      # run the execubable.
  y = rt.get_var('y')              # get the executable from spu runtime.
```


| Field | Type | Description |
| ----- | ---- | ----------- |
| name | [ string](#string) | The name of the executable. |
| input_names | [repeated string](#string) | The input names. |
| output_names | [repeated string](#string) | The output names. |
| code | [ bytes](#bytes) | The bytecode of the program, with format IR_MLIR_SPU. |
 <!-- end Fields -->
 <!-- end HasFields -->


### IrProto
The immediate representation proto.


| Field | Type | Description |
| ----- | ---- | ----------- |
| ir_type | [ IrType](#irtype) | The IR type. |
| code | [ bytes](#bytes) | Code format is defined by IrType. |
| meta | [ XlaMeta](#xlameta) | Only meaningful for IR_XLA_HLO |
 <!-- end Fields -->
 <!-- end HasFields -->


### RuntimeConfig
The SPU runtime configuration.


| Field | Type | Description |
| ----- | ---- | ----------- |
| protocol | [ ProtocolKind](#protocolkind) | The protocol kind. |
| field | [ FieldType](#fieldtype) | The field type. |
| fxp_fraction_bits | [ int64](#int64) | Number of fraction bits of fixed-point number. |
| enable_action_trace | [ bool](#bool) | When enabled, runtime prints verbose info of the callstack, debug purpose only. |
| enable_type_checker | [ bool](#bool) | When enabled, runtime checks runtime type infos against the compile-time ones, exceptions are raised if mismatches happen. Note: Runtime outputs prefer runtime type infos even when flag is on. |
| enable_pphlo_trace | [ bool](#bool) | When enabled, runtime prints executed pphlo list, debug purpose only. |
| enable_processor_dump | [ bool](#bool) | When enabled, runtime dumps executed executables in the dump_dir, debug purpose only. |
| processor_dump_dir | [ string](#string) | none |
| enable_pphlo_profile | [ bool](#bool) | When enabled, runtime records detailed pphlo timing data, debug purpose only. |
| enable_hal_profile | [ bool](#bool) | When enabled, runtime records detailed hal timing data, debug purpose only. |
| reveal_secret_condition | [ bool](#bool) | Allow runtime to reveal `secret variable` use as if and while condition result, debug purpose only. |
| reveal_secret_indicies | [ bool](#bool) | Allow runtime to reveal `secret variable` use as indices, debug purpose only. |
| public_random_seed | [ uint64](#uint64) | The public random variable generated by the runtime, the concrete prg function is implementation defined. Note: this seed only applies to `public variable` only, it has nothing to do with security. |
| fxp_div_goldschmidt_iters | [ int64](#int64) | The iterations use in f_div with Goldschmidt method. 0(default) indicates implementation defined. |
| fxp_exp_mode | [ RuntimeConfig.ExpMode](#runtimeconfigexpmode) | The exponent approximation method. |
| fxp_exp_iters | [ int64](#int64) | Number of iterations of `exp` approximation, 0(default) indicates impl defined. |
| fxp_log_mode | [ RuntimeConfig.LogMode](#runtimeconfiglogmode) | The logarithm approximation method. |
| fxp_log_iters | [ int64](#int64) | Number of iterations of `log` approximation, 0(default) indicates impl-defined. |
| fxp_log_orders | [ int64](#int64) | Number of orders of `log` approximation, 0(default) indicates impl defined. |
| sigmoid_mode | [ RuntimeConfig.SigmoidMode](#runtimeconfigsigmoidmode) | The sigmoid function approximation model. |
 <!-- end Fields -->
 <!-- end HasFields -->


### ShapeProto



| Field | Type | Description |
| ----- | ---- | ----------- |
| dims | [repeated int64](#int64) | none |
 <!-- end Fields -->
 <!-- end HasFields -->


### ValueProto
The spu Value proto, used for spu value serialization.


| Field | Type | Description |
| ----- | ---- | ----------- |
| data_type | [ DataType](#datatype) | The data type. |
| visibility | [ Visibility](#visibility) | The data visibility. |
| shape | [ ShapeProto](#shapeproto) | The shape of the value. |
| storage_type | [ string](#string) | The storage type, defined by the underline evaluation engine. i.e. `aby3.AShr<FM64>` means an aby3 arithmetic share in FM64. usually, the application does not care about this attribute. |
| content | [ bytes](#bytes) | The runtime/protocol dependent value data. |
 <!-- end Fields -->
 <!-- end HasFields -->


### XlaMeta



| Field | Type | Description |
| ----- | ---- | ----------- |
| inputs | [repeated Visibility](#visibility) | none |
 <!-- end Fields -->
 <!-- end HasFields -->
 <!-- end messages -->

## Enums


### DataType
The SPU datatype

| Name | Number | Description |
| ---- | ------ | ----------- |
| DT_INVALID | 0 | none |
| DT_I1 | 1 | 1bit integer (bool). |
| DT_I8 | 2 | int8 |
| DT_U8 | 3 | uint8 |
| DT_I16 | 4 | int16 |
| DT_U16 | 5 | uint16 |
| DT_I32 | 6 | int32 |
| DT_U32 | 7 | uint32 |
| DT_I64 | 8 | int64 |
| DT_U64 | 9 | uint64 |
| DT_FXP | 10 | Fixed point type. |




### FieldType
A security parameter type.

The secure evaluation is based on some algebraic structure (ring or field),

| Name | Number | Description |
| ---- | ------ | ----------- |
| FT_INVALID | 0 | none |
| FM32 | 1 | Ring 2^32 |
| FM64 | 2 | Ring 2^64 |
| FM128 | 3 | Ring 2^128 |




### IrType
The immediate representation type.

| Name | Number | Description |
| ---- | ------ | ----------- |
| IR_INVALID | 0 | none |
| IR_XLA_HLO | 1 | IR_XLA_HLO means the code part of IrProto is XLA protobuf binary format. See https://www.tensorflow.org/xla/architecture for details. |
| IR_MLIR_SPU | 2 | IR_MLIR_SPU means the code part of IrProto is pphlo MLIR text format. See spu/dialect/pphlo_dialect.td for details. |




### ProtocolKind
The protocol kind.

| Name | Number | Description |
| ---- | ------ | ----------- |
| PROT_INVALID | 0 | Invalid protocol. |
| REF2K | 1 | The reference implementation in `ring^2k`, note: this 'protocol' only behave-like a fixed point secure protocol without any security guarantee. Hence, it should only be selected for debugging purposes. |
| SEMI2K | 2 | A semi-honest multi-party protocol. This protocol requires a trusted third party to generate the offline correlated randoms. Currently, Secretflow by default ship this protocol with a trusted first party. Hence, it should only be used for debugging purposes. |
| ABY3 | 3 | A honest majority 3PC-protocol. Secretflow provides the semi-honest implementation without Yao. |
| CHEETAH | 4 | The famous [Cheetah](https://eprint.iacr.org/2022/207) protocol, a very fast 2PC protocol. |




### PtType
Plaintext type

SPU runtime does not process with plaintext directly, plaintext type is
mainly used for IO purposes, when converting a plaintext buffer to an SPU
buffer, we have to let spu know which type the plaintext buffer is.

| Name | Number | Description |
| ---- | ------ | ----------- |
| PT_INVALID | 0 | none |
| PT_I8 | 1 | int8_t |
| PT_U8 | 2 | uint8_t |
| PT_I16 | 3 | int16_t |
| PT_U16 | 4 | uint16_t |
| PT_I32 | 5 | int32_t |
| PT_U32 | 6 | uint32_t |
| PT_I64 | 7 | int64_t |
| PT_U64 | 8 | uint64_t |
| PT_F32 | 9 | float |
| PT_F64 | 10 | double |
| PT_I128 | 11 | int128_t |
| PT_U128 | 12 | uint128_t |
| PT_BOOL | 13 | bool |




### RuntimeConfig.ExpMode
The exponential approximation method.

| Name | Number | Description |
| ---- | ------ | ----------- |
| EXP_DEFAULT | 0 | Implementation defined. |
| EXP_PADE | 1 | The pade approximation. |
| EXP_TAYLOR | 2 | Taylor series approximation. |




### RuntimeConfig.LogMode
The logarithm approximation method.

| Name | Number | Description |
| ---- | ------ | ----------- |
| LOG_DEFAULT | 0 | Implementation defined. |
| LOG_PADE | 1 | The pade approximation. |
| LOG_NEWTON | 2 | The newton approximation. |




### RuntimeConfig.SigmoidMode
The sigmoid approximation method.

| Name | Number | Description |
| ---- | ------ | ----------- |
| SIGMOID_DEFAULT | 0 | Implementation defined. |
| SIGMOID_MM1 | 1 | Minmax approximation one order. f(x) = 0.5 + 0.125 * x |
| SIGMOID_SEG3 | 2 | Piece-wise simulation. f(x) = 0.5 + 0.125x if -4 <= x <= 4 1 if x > 4 0 if -4 > x |
| SIGMOID_REAL | 3 | The real definition, which depends on exp's accuracy. f(x) = 1 / (1 + exp(-x)) |




### Visibility
The visibility type.

SPU is secure evaluation runtime, but not all data are secret, some of them
are publicly known to all parties, mark them as public will improve
performance significantly.

| Name | Number | Description |
| ---- | ------ | ----------- |
| VIS_INVALID | 0 | none |
| VIS_SECRET | 1 | Invisible(unknown) for all or some of the parties. |
| VIS_PUBLIC | 2 | Visible(public) for all parties. |


 <!-- end Enums -->
 <!-- end Files -->

## Scalar Value Types

| .proto Type | Notes | C++ Type | Java Type | Python Type |
| ----------- | ----- | -------- | --------- | ----------- |
| <div><h4 id="double" /></div><a name="double" /> double |  | double | double | float |
| <div><h4 id="float" /></div><a name="float" /> float |  | float | float | float |
| <div><h4 id="int32" /></div><a name="int32" /> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int | int |
| <div><h4 id="int64" /></div><a name="int64" /> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | long | int/long |
| <div><h4 id="uint32" /></div><a name="uint32" /> uint32 | Uses variable-length encoding. | uint32 | int | int/long |
| <div><h4 id="uint64" /></div><a name="uint64" /> uint64 | Uses variable-length encoding. | uint64 | long | int/long |
| <div><h4 id="sint32" /></div><a name="sint32" /> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int | int |
| <div><h4 id="sint64" /></div><a name="sint64" /> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | long | int/long |
| <div><h4 id="fixed32" /></div><a name="fixed32" /> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int | int |
| <div><h4 id="fixed64" /></div><a name="fixed64" /> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | long | int/long |
| <div><h4 id="sfixed32" /></div><a name="sfixed32" /> sfixed32 | Always four bytes. | int32 | int | int |
| <div><h4 id="sfixed64" /></div><a name="sfixed64" /> sfixed64 | Always eight bytes. | int64 | long | int/long |
| <div><h4 id="bool" /></div><a name="bool" /> bool |  | bool | boolean | boolean |
| <div><h4 id="string" /></div><a name="string" /> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | String | str/unicode |
| <div><h4 id="bytes" /></div><a name="bytes" /> bytes | May contain any arbitrary sequence of bytes. | string | ByteString | str |
