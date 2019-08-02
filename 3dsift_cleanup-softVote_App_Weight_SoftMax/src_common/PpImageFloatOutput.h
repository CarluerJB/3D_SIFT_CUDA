
#ifndef __PPIMAGEFLOATOUTPUT_H__
#define __PPIMAGEFLOATOUTPUT_H__

#include "PpImage.h"

//
// output_text()
//
// Output image as a text file.
//
void
output_float_to_text(
	const PpImage &ppImgFloat,
	char *pcFName,
	int iFeature = 0
	);

void
output_float(
	const PpImage &ppImg,
	char *pcFName,
	int iFeature = 0
	);

void
output_float(
	const PpImage &ppImgFloat,
	PpImage &ppImgChar,
	int iFeature = 0
	);

void
output_int(
	const PpImage &ppImg,
	char *pcFName
	);

void
output_float_clip(
	const PpImage &ppImg,
	char *pcFName,
	const float fLower,
	const float fUpper,
	int iFeature = 0
	);

#endif

