
define weak_odr <16 x i16>  @packssdwx16(<16 x i32> %arg) nounwind alwaysinline {
  %1 = shufflevector <16 x i32> %arg, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %2 = shufflevector <16 x i32> %arg, <16 x i32> poison, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %3 = tail call <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32> %1, <8 x i32> %2)
  ret <16 x i16> %3
}
declare <16 x i16> @llvm.x86.avx2.packssdw(<8 x i32>, <8 x i32>)

define weak_odr <32 x i8> @packuswbx32(<32 x i16> %arg) nounwind alwaysinline {
 %1 = shufflevector <32 x i16> %arg, <32 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
 %2 = shufflevector <32 x i16> %arg, <32 x i16> poison, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
 %3 = call <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %1, <16 x i16> %2)
 ret <32 x i8> %3
}
declare <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16>, <16 x i16>)

define weak_odr <32 x i8> @packsswbx32(<32 x i16> %arg) nounwind alwaysinline {
 %1 = shufflevector <32 x i16> %arg, <32 x i16> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
 %2 = shufflevector <32 x i16> %arg, <32 x i16> poison, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
 %3 = call <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16> %1, <16 x i16> %2)
 ret <32 x i8> %3
}
declare <32 x i8> @llvm.x86.avx2.packsswb(<16 x i16>, <16 x i16>)

define weak_odr <16 x i16>  @packusdwx16(<16 x i32> %arg) nounwind alwaysinline {
  %1 = shufflevector <16 x i32> %arg, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %2 = shufflevector <16 x i32> %arg, <16 x i32> poison, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %3 = tail call <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32> %1, <8 x i32> %2)
  ret <16 x i16> %3
}
declare <16 x i16> @llvm.x86.avx2.packusdw(<8 x i32>, <8 x i32>) nounwind readnone

define weak_odr <32 x i8> @abs_i8x32(<32 x i8> %arg) {
 %1 = sub <32 x i8> zeroinitializer, %arg
 %2 = icmp sgt <32 x i8> %arg, zeroinitializer
 %3 = select <32 x i1> %2, <32 x i8> %arg, <32 x i8> %1
 ret <32 x i8> %3
}

define weak_odr <16 x i16> @abs_i16x16(<16 x i16> %arg) {
 %1 = sub <16 x i16> zeroinitializer, %arg
 %2 = icmp sgt <16 x i16> %arg, zeroinitializer
 %3 = select <16 x i1> %2, <16 x i16> %arg, <16 x i16> %1
 ret <16 x i16> %3
}

define weak_odr <8 x i32> @abs_i32x8(<8 x i32> %arg) {
 %1 = sub <8 x i32> zeroinitializer, %arg
 %2 = icmp sgt <8 x i32> %arg, zeroinitializer
 %3 = select <8 x i1> %2, <8 x i32> %arg, <8 x i32> %1
 ret <8 x i32> %3
}

define weak_odr <16 x i16> @saturating_pmulhrswx16(<16 x i16> %a, <16 x i16> %b) nounwind uwtable readnone alwaysinline {
  %1 = tail call <16 x i16> @llvm.x86.avx2.pmul.hr.sw(<16 x i16> %a, <16 x i16> %b)
  %2 = icmp eq <16 x i16> %a, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %3 = icmp eq <16 x i16> %b, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %4 = and <16 x i1> %2, %3
  %5 = select <16 x i1> %4, <16 x i16> <i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767>, <16 x i16> %1
  ret <16 x i16> %5
}
declare <16 x i16> @llvm.x86.avx2.pmul.hr.sw(<16 x i16>, <16 x i16>) nounwind readnone

define weak_odr <16 x i16> @hadd_pmadd_u8_avx2(<32 x i8> %a) nounwind alwaysinline {
  %1 = tail call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> %a, <32 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  ret <16 x i16> %1
}

define weak_odr <16 x i16> @hadd_pmadd_i8_avx2(<32 x i8> %a) nounwind alwaysinline {
  %1 = tail call <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <32 x i8> %a)
  ret <16 x i16> %1
}
declare <16 x i16> @llvm.x86.avx2.pmadd.ub.sw(<32 x i8>, <32 x i8>) nounwind readnone
