#include <iostream>
#include <thread>

#include "nifti1_io.h"
#include "FeatureIOnifti.h"
#include "featMatchUtilities.h"
#include "TextFile.h"

#define MAX_CORES 32


//
// matchAllToAll()
//
// Match all images to all images.
//
int
matchAllToAll(
	vector< char * > &vecNames,
	vector< int > &vecMatched,
	vector< vector<Feature3DInfo> > &vvInputFeats,
	int iNeighbors,	// number of nearest neighbors to consider
	vector< int > &viLabels, // labels for each feature set
	int bOnlyPeaksFeatures,
	char* peakOrValley,
	float fGeometryWeight = -1,
	char *pcOutputFile = 0,
	int iImgSplit = -1  // image to split on for testing
)
{
	printf("Creating NN index structure, NN=%d, image split=%d, type features=%s\n", iNeighbors, iImgSplit, peakOrValley);

	msNearestNeighborApproximateInit(vvInputFeats, iNeighbors, viLabels, fGeometryWeight, iImgSplit);

	printf("done.\n");

	vector< int > viImgCounts;
	vector< float > viLabelCounts;
	vector< float > vfLabelLogLikelihood;

	// Count labels
	viImgCounts.resize(vecNames.size(), 0);

	// Assume labels go from
	int iMaxLabel = 0;
	for (int i = 0; i < viLabels.size(); i++)
	{
		if (viLabels[i] > iMaxLabel)
		{
			iMaxLabel = viLabels[i];
		}
	}
	viLabelCounts.resize(iMaxLabel + 1);
	vfLabelLogLikelihood.resize(iMaxLabel + 1);
	FILE *outfileVotes;
	FILE* outfileVoteCount;
	// Store votes
	if (bOnlyPeaksFeatures==2) {
		outfileVotes = fopen("matching_votes.txt", "at");
		outfileVoteCount = fopen("vote_count.txt", "at");
	}
	else{
		outfileVotes = fopen("matching_votes.txt", "wt");
		outfileVoteCount = fopen("vote_count.txt", "wt");
	}

	float **ppfMatchingVotes = new float*[vvInputFeats.size()];
	int **ppiLabelVotes = new int*[vvInputFeats.size()];
	for (int i = 0; i < vvInputFeats.size(); i++)
	{
		ppfMatchingVotes[i] = new float[vvInputFeats.size()];
		ppiLabelVotes[i] = new int[vvInputFeats.size()];
		for (int j = 0; j < vvInputFeats.size(); j++)
		{
			ppfMatchingVotes[i][j] = 0.0;
			ppiLabelVotes[i][j] = 0;
		}
	}

	// Try max core detection with std::thread::hardware_concurrency() (C++11)
	int maxCores = std::thread::hardware_concurrency();
	if (maxCores <= 0 || maxCores > MAX_CORES) {
		maxCores = MAX_CORES;
	}
	int nCores = vvInputFeats.size() < maxCores ? vvInputFeats.size() : maxCores;

	// Create image chunks
	int iChunkSize = std::ceil(float(vvInputFeats.size()) / nCores);

	std::vector< std::pair<int,int> > vpChunkStartEnd;
	for (int i = 0; i < nCores; ++i)
	{
		int iStart = i*iChunkSize;
		int iEnd = 0;

		if (i == nCores-1)
		{
			iEnd = vvInputFeats.size()-1;
		}
		else
		{
			iEnd = (i+1)*iChunkSize-1;
		}
		std::pair<int,int> pStartEnd(iStart, iEnd);
		vpChunkStartEnd.push_back(pStartEnd);
	}

	#pragma omp parallel for num_threads(nCores) schedule(static,1)
	for (int n = 0; n < nCores; ++n)
	{
		for (int i = vpChunkStartEnd[n].first; i <= vpChunkStartEnd[n].second; i++)
		{
			printf("Searching image %d of %d ... ", i, vvInputFeats.size());
			msNearestNeighborApproximateSearchSelf(i, viImgCounts, viLabelCounts, vfLabelLogLikelihood, ppfMatchingVotes, ppiLabelVotes);
			printf("\n");
		}
	}

	// Output votes
	fprintf(outfileVotes, "%s\n", peakOrValley);
	fprintf(outfileVoteCount, "%s\n", peakOrValley);
	for (int i = 0; i < vvInputFeats.size(); i++)
	{
		for (int j = 0; j < vvInputFeats.size(); j++)
		{
			float vote = (i == j) ? 0.0 : ppfMatchingVotes[i][j] / ppiLabelVotes[i][j];


			fprintf(outfileVotes, "%f\t", ppfMatchingVotes[i][j]);
			fprintf(outfileVoteCount, "%d\t", ppiLabelVotes[i][j]);
		}
		fprintf(outfileVotes, "\n");
		fprintf(outfileVoteCount, "\n");
		delete[] ppfMatchingVotes[i];
	}
	delete[] ppfMatchingVotes;
	fprintf(outfileVotes, "\n");
	fprintf(outfileVoteCount, "\n");
	fclose(outfileVotes);
	fclose(outfileVoteCount);

	msNearestNeighborApproximateDelete();

	return 0;
}

int
matchAllToOne(
	vector< char * > &vecNames,
	vector< int > &vecMatched,
	vector< vector<Feature3DInfo> > &vvInputFeats,
	int bExpandedMatching = 0, // Expanded matching, try to find some additional correspondences (inliers of inliers)
	char *pcOutFileName = 0, // Output file name, for results, etc
	int bSequential = 0, // Sequential matching
	int bOutputFeatures = 0 // Visualization
)
{
	vector<int> vecModelMatches;
	vector<Feature3DInfo> vecFeats2Updated;
	vector<int> vecModelMatchesUpdated;

	TransformSimilarity ts12, ts12Next, ts12Inv, ts31, ts32;

	int iImg1=-1, iImg2=-1, iImg3=-1;

	int bRefine = 0;

	// Model features for inspection, RIRE 39
	int piFeatsToMatch[] = {7, 129, 1142, 11, 12, 25, 29, 299, 1086, 905, 1266, 304, 2, 69, 76, 88, 89, 93, 221, 341, 1081, 1149, 1357, 0};
	int iFeatsToMatchSize = sizeof(piFeatsToMatch)/sizeof(int);
	vector<int> vecFeatsToMatchFound;
	vecFeatsToMatchFound.resize( vecNames.size(), 0 );
	vector<int> vecFeatsToMatchFreq;
	vecFeatsToMatchFreq.resize( vvInputFeats[0].size(), 0 );

	// Keep track of matches - ahh, do this in learning ...
	int iImage0Feats = vvInputFeats[0].size();
	unsigned char *pucMatches = new unsigned char[iImage0Feats*vvInputFeats.size()];
	memset( pucMatches, 0, vvInputFeats[0].size()*vvInputFeats.size() );

	//while( iImg1 == -1 )
	//{
	// iImg1 = (rand()*iFeatVec) / RAND_MAX;
	//}
	iImg1 = 0;
	for( int i = 0; i < vvInputFeats.size(); i++ )
	{
		if( i == iImg1 )
		{
			continue;
		}

		if( bSequential )
		{
			// Match in sequence

			iImg1 = i;
			iImg2 = i+1;
			if( i+1 == vvInputFeats.size() )
			{
				iImg2 = 0; // wrap around? for cardiac data
				//continue;
			}
		}
		else
		{
			iImg2 = i;
		}

		//iImg2 = 0;
		//iImg1 = i;

		char pcImg1[400], pcImg2[400];
		sprintf( pcImg1, "%s", vecNames[iImg1] );
		sprintf( pcImg2, "%s", vecNames[iImg2] );
		char *pch;

		pch = strrchr( pcImg1, '.' );
		//pch[1] = 'i';pch[2] = 'm';pch[3] = 'g';
		//sprintf( pch, ".nii.gz" );
		sprintf( pch, ".hdr" );

		pch = strrchr( pcImg2, '.' );
		//pch[1] = 'i';pch[2] = 'm';pch[3] = 'g';
		//sprintf( pch, ".nii.gz" );
		sprintf( pch, ".hdr" );

		char *pci1=0;
		char *pci2=0;

		if( bOutputFeatures )
		{
			pci1=pcImg1; pci2=pcImg2;
		}

		// Match 1->2
		int iInliers1 = MatchKeys( vvInputFeats[iImg1], vvInputFeats[iImg2], vecModelMatches,
			-1, pci1, pci2, &vecFeats2Updated, bExpandedMatching, &ts12.m_fScale, ts12.m_ppfRot[0], ts12.m_pfTrans,
			0 );
			//bExpandedMatching );
			int iInliers2 = 0;
			int iInliers3 = 0;
			int iMatches;

			// This statement is for short circuiting
			// printf( "Inliers: %d\n", iInliers1 ); continue;

			iMatches = 0;
			for( int g = 0; g < vecModelMatches.size(); g++ )
			{
				if( vecModelMatches[g] >= 0 )
				{
					iMatches++;
					if( pucMatches )
					{
						pucMatches[iImg2*iImage0Feats + g] = vecModelMatches[g];
					}

					// Evaluate coverage for set of ground truth matches
					for( int j = 0; j < iFeatsToMatchSize; j++ )
					{
						if( g == piFeatsToMatch[j] )
						{
							vecFeatsToMatchFound[iImg2]++;
						}
					}

					if( !bSequential ) vecFeatsToMatchFreq[g]++;
				}
			}

			char pcFileName[400];
			FILE *outfile;
			FILE *outInfoFile;

			//
			// This is old output
			//
			//sprintf( pcFileName, "%s.matches.txt", vecNames[i] );
			//outfile = fopen( pcFileName, "wt" );
			//pci1 = strrchr( pcImg1, '\\' ) + 1;
			//pci2 = strrchr( pcImg2, '\\' ) + 1;
			//fprintf( outfile, "# Img1: %s\n", pcImg1 );
			//fprintf( outfile, "# Img2: %s\n", pcImg2 );
			//fprintf( outfile, "# Matches: %d\n", iMatches );
			//fprintf( outfile, "# Format: x1 y1 z1 s1 x2 y2 z2 s2 SquaredDifference\n" );
			//for( int g = 0; g < vecModelMatches.size(); g++ )
			//{
			// if( vecModelMatches[g] >= 0 )
			// {
			// Feature3DInfo &f1 = vvInputFeats[iImg1][g];
			// Feature3DInfo &f2 = vvInputFeats[iImg2][ vecModelMatches[g] ];
			// float fDistSqr = f1.DistSqrPCs( f2 );
			// fprintf( outfile, "%f\t%f\t%f\t%f\t", f1.x, f1.y, f1.z, f1.scale);
			// fprintf( outfile, "%f\t%f\t%f\t%f\t", f2.x, f2.y, f2.z, f2.scale);
			// fprintf( outfile, "%f\n", fDistSqr );
			// }
			//}
			//fclose( outfile );

			sprintf( pcFileName, "%s.matches.info.txt", vecNames[i] );
			outInfoFile = fopen( pcFileName, "wt" );

			sprintf( pcFileName, "%s.matches.img1.txt", vecNames[i] );
			outfile = fopen( pcFileName, "wt" );
			fprintf( outfile, "# Img1: %s\n", pcImg1 );
			fprintf( outfile, "# Img2: %s\n", pcImg2 );
			fprintf( outfile, "# Matches: %d\n", iMatches );
			fprintf( outfile, "# Format: Img1 x1 y1 z1 s1 MatchIndexImg2 DistSqr\n" );
			int iCurrMatch = 0;
			for( int g = 0; g < vecModelMatches.size(); g++ )
			{
				if( vecModelMatches[g] >= 0 )
				{
					Feature3DInfo &f1 = vvInputFeats[iImg1][g];
					Feature3DInfo &f2 = vvInputFeats[iImg2][ vecModelMatches[g] ];
					float fDistSqr = 0; //f1.DistSqrPCs( f2 );
					fprintf( outInfoFile, "%d\t%d\n", f1.m_uiInfo, f2.m_uiInfo );

					//fprintf( outfile, "%s\t%f\t%f\t%f\t%f\timg2_match%4.4d_feat%6.6d\t%f\n", vecNames[iImg1], f1.x, f1.y, f1.z, f1.scale, iCurrMatch, vecModelMatches[g], fDistSqr );
					fprintf( outfile, "%s\t%f\t%f\t%f\t%f\timg2_match%4.4d_feat%6.6d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", vecNames[iImg1], f1.x, f1.y, f1.z, f1.scale, iCurrMatch, vecModelMatches[g], fDistSqr,
					f1.ori[0][0], f1.ori[0][1], f1.ori[0][2],
					f1.ori[1][0], f1.ori[1][1], f1.ori[1][2],
					f1.ori[2][0], f1.ori[2][1], f1.ori[2][2]
				);
				iCurrMatch++;
			}
		}
		fclose( outfile );
		fclose( outInfoFile );

		sprintf( pcFileName, "%s.matches.img2.txt", vecNames[i] );
		outfile = fopen( pcFileName, "wt" );
		fprintf( outfile, "# Img1: %s\n", pcImg1 );
		fprintf( outfile, "# Img2: %s\n", pcImg2 );
		fprintf( outfile, "# Matches: %d\n", iMatches );
		fprintf( outfile, "# Format: Img2 x2 y2 z2 s2 MatchIndexImg1 DistSqr\n" );
		iCurrMatch = 0;
		for( int g = 0; g < vecModelMatches.size(); g++ )
		{
			if( vecModelMatches[g] >= 0 )
			{
				Feature3DInfo &f1 = vvInputFeats[iImg1][g];
				Feature3DInfo &f2 = vvInputFeats[iImg2][ vecModelMatches[g] ];
				float fDistSqr = 0; //f1.DistSqrPCs( f2 );
				//fprintf( outfile, "%s\t%f\t%f\t%f\t%f\timg1_match%4.4d_feat%6.6d\t%f\n", vecNames[iImg2], f2.x, f2.y, f2.z, f2.scale, iCurrMatch, g, fDistSqr );
				fprintf( outfile, "%s\t%f\t%f\t%f\t%f\timg2_match%4.4d_feat%6.6d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", vecNames[iImg2], f2.x, f2.y, f2.z, f2.scale, iCurrMatch, g, fDistSqr,
				f2.ori[0][0], f2.ori[0][1], f2.ori[0][2],
				f2.ori[1][0], f2.ori[1][1], f2.ori[1][2],
				f2.ori[2][0], f2.ori[2][1], f2.ori[2][2]
			);

			iCurrMatch++;
		}
	}
	fclose( outfile );


	sprintf( pcFileName, "%s.trans.txt", vecNames[i] );
	ts12.WriteMatrix( pcFileName );

	ts12Inv = ts12;
	ts12Inv.Invert();
	sprintf( pcFileName, "%s.trans-inverse.txt", vecNames[i] );
	ts12Inv.WriteMatrix( pcFileName );

	//ts12.Multiply( ts12Inv );
	//sprintf( pcFileName, "%s.zero.txt", vecNames[i] );
	//ts12.WriteMatrix( pcFileName );
	printf( "%s: inliers %d\t%d\t%d\t%f\n", vecNames[i],
	iInliers1, iInliers2, iInliers3, ts12.m_fScale );

	if( pcOutFileName )
	{
		//FILE *outfileR = fopen( "report.txt", "a+" );
		FILE *outfileR = fopen( pcOutFileName, "a+" );
		fprintf( outfileR, "%s:\tinliers\t%d\t%d\t%d\t%f\t%f\t%f\t%f\n", vecNames[i],
		iInliers1, iInliers2, iInliers3, ts12.m_fScale, ts12.m_pfTrans[0], ts12.m_pfTrans[1], ts12.m_pfTrans[2] );
		fclose( outfileR );
	}

	// Transform features in image 2 to image 1 space, output aligned features
	for( int g = 0; g < vvInputFeats[iImg2].size(); g++ )
	{
		float pfZero[3] = {0,0,0};
		//vvInputFeats[iImg2][g].SimilarityTransform( pfZero, ts12.m_pfTrans, ts12.m_ppfRot[0], ts12.m_fScale );
	}
	sprintf( pcFileName, "%s.update.key", vecNames[i] );
	msFeature3DVectorOutputText( vvInputFeats[iImg2], pcFileName );
}

printf( "\n" );

return 1;
}


int main(int argc, char **argv)
{
	if (argc < 3)
	{
		printf("Volumetric Feature matching v1.1\n");
		printf("Determines robust alignment solution mapping coordinates in image 2, 3, ... to image 1.\n");
		printf("Usage: %s [options] <input keys 1> <input keys 2> ... \n", "featMatchMultiple");
		printf("  <input keys 1, ...>: input key files, produced from featExtract.\n");
		printf("  <output transform>: output text file with linear transform from keys 2 -> keys 1.\n");
		return -1;
	}


	// Output command in file for log purpose
	FILE *commandFile = fopen("_command.txt", "wt");
	for (int i = 0; i < argc; ++i)
	{
		fprintf(commandFile, "%s ", argv[i]);
	}
	fprintf(commandFile,"\n");
	fclose(commandFile);

	int iArg = 1;
	int bMultiModal = 0;
	int bExpandedMatching = 0;
	int bOnlyReorientedFeatures = 1;
	int bOnlyPeaksFeatures = 4;
	char *pcOutputFileName = "report.txt";
	char *pcInputFileList = 0;
	char *pcLabelFile = 0;
	int iMatchType = 2;
	int bViewFeatures = 0;
	int iNeighbors = 5;
	float fGeometryWeight = -1;
	float iImgSplit = -1;
	char a;
	while (iArg < argc && argv[iArg][0] == '-')
	{
		switch (argv[iArg][1])
		{
			case 'o': case 'O':
			// General output file name
			iArg++;
			pcOutputFileName = argv[iArg];
			iArg++;
			break;

			case 's': case 'S':
			// Option to deal with Valleys and Peaks, split peak and valley could increase accuracy
			bOnlyPeaksFeatures =  stoi(&argv[iArg][2]);
			iArg++;
			break;

			case 'r': case 'R':
			// Option to not use reoriented features - for volumes that are already aligned
			bOnlyReorientedFeatures = 1;
			if (argv[iArg][2] == '-')
			{
				// Only
				bOnlyReorientedFeatures = 0;
			}
			iArg++;
			break;

			case 'n': case 'N':

			//
			// Number of nearest neighbors to consider
			//
			iArg++;
			iNeighbors = atoi(argv[iArg]);
			iArg++;
			break;

			case 'f': case 'F':
			//
			// Input features are listed in a single file (for long file lists)
			//
			iArg++;
			pcInputFileList = argv[iArg];
			iArg++;
			break;

			default:
			printf("Error: unknown command line argument: %s\n", argv[iArg]);
			return -1;
			break;
		}
	}

	FILE *outfileR = fopen(pcOutputFileName, "wt");
	fclose(outfileR);

	vector< char * > vecNames;
	vector< int > vLabels;
	int iTotalFeats = 0;
	int iFeatVec = 0;
	int iFeatVecTotal = 0;

	// Create file list for read in - allows input file, for long lists of files
	// for example, longer than the linux limit: >> getconf ARG_MAX
	TextFile tfNames;
	if (pcInputFileList)
	{
		if (tfNames.Read(pcInputFileList) != 0)
		{
			printf("Error: could not read input file name list: %s\n");
			return -1;
		}
		iFeatVec = 0;
		for (int i = 0; i < tfNames.Lines(); i++)
		{
			if (strlen(tfNames.GetLine(i)) > 0)
			{
				vecNames.push_back(tfNames.GetLine(i));
				iFeatVec++;
			}
		}
		iFeatVecTotal = iFeatVec;
	}
	else
	{
		iFeatVec = 0;
		iFeatVecTotal = argc - iArg;
		for (int i = iArg; i < argc; i++)
		{
			vecNames.push_back(argv[i]);
			iFeatVec++;
		}
		assert(iFeatVec == iFeatVecTotal);
	}

	// read in labels
	TextFile tfLabels;
	if (pcLabelFile)
	{
		if (tfLabels.Read(pcLabelFile) != 0)
		{
			printf("Error: could not read input file name list: %s\n");
			return -1;
		}
		iFeatVec = 0;
		for (int i = 0; i < tfLabels.Lines(); i++)
		{
			if (strlen(tfLabels.GetLine(i)) > 0)
			{
				int iLabel = atoi(tfLabels.GetLine(i));
				vLabels.push_back(iLabel);
				iFeatVec++;
			}
		}
		assert(vLabels.size() == vecNames.size());
	}
	else
	{
		// Default labels = simply image indices
		vLabels.resize(vecNames.size());
		for (int i = 0; i < vLabels.size(); i++)
		{
			vLabels[i] = i;
		}
	}

	FILE *outfileNames = fopen("_names.txt", "wt");
	for (iFeatVec = 0; iFeatVec < iFeatVecTotal; iFeatVec++)
	{
		fprintf(outfileNames, "%s\t%d\n", vecNames[iFeatVec], vLabels[iFeatVec]);
	}
	fclose(outfileNames);

	assert(vecNames.size() == iFeatVecTotal);

	vector< int > vecMatched;
	vector< vector<Feature3DInfo> > vvInputFeats;
	vecMatched.resize(vecNames.size(), -1);
	vvInputFeats.resize(vecNames.size());
	vector<vector< vector<Feature3DInfo> >> SplitedFeat(2);
	SplitedFeat[0].resize(vecNames.size());
	SplitedFeat[1].resize(vecNames.size());
	char* FeatType = "Peak and Valley";
	for (iFeatVec = 0; iFeatVec < iFeatVecTotal; iFeatVec++)
	{
		char pcImg1[400];
		sprintf(pcImg1, "%s", vecNames[iFeatVec]);
		char *pch = strrchr(pcImg1, '\\');
		if (pch) pch++; else pch = pcImg1;

		int bSameName = 0;
		for (int j = 0; j < iFeatVec && bSameName == 0; j++)
		{
			if (strcmp(vecNames[iFeatVec], vecNames[j]) == 0)
			{
				bSameName = 1;
			}
		}

		printf("Reading file %d: %s...", iFeatVec, pch);

		if (msFeature3DVectorInputText(vvInputFeats[iFeatVec], vecNames[iFeatVec], 140) < 0)
		{
			printf("Error: could not open feature file %d: %s\n", iFeatVec, vecNames[iFeatVec]);
			continue;
		}

		if (bOnlyReorientedFeatures)
		{
			removeNonReorientedFeatures(vvInputFeats[iFeatVec]);
		}
		else
		{
			removeReorientedFeatures(vvInputFeats[iFeatVec]);
		}

		if (bOnlyPeaksFeatures==0) {
			removeNonPeakFeatures(vvInputFeats[iFeatVec]);
			FeatType = "Peaks";
		}

		if (bOnlyPeaksFeatures==1) {
			removeNonValleyFeatures(vvInputFeats[iFeatVec]);
			FeatType = "Valley";
		}
		if (bOnlyPeaksFeatures==2) {
			(SplitedFeat[0][iFeatVec])=(vvInputFeats[iFeatVec]);
			(SplitedFeat[1][iFeatVec])=(vvInputFeats[iFeatVec]);
			SplitFeatures(SplitedFeat, iFeatVec);
		}

		iTotalFeats += vvInputFeats[iFeatVec].size();
		printf("feats: %d, total: %d\n", vvInputFeats[iFeatVec].size(), iTotalFeats);
	}

	vecNames.resize(iFeatVec);
	vvInputFeats.resize(iFeatVec);
	vecMatched.resize(iFeatVec, -1);

	FILE *outfile = fopen("feature_count.txt", "wt");
	for (int i = 0; i < vvInputFeats.size(); i++)
	{
		fprintf(outfile, "%d\t%d\n", i, vvInputFeats[i].size());
	}
	fclose(outfile);
	matchAllToOne(vecNames, vecMatched, vvInputFeats);
	if (bOnlyPeaksFeatures==2) {
		matchAllToOne(vecNames, vecMatched, SplitedFeat[0]);
		matchAllToOne(vecNames, vecMatched, SplitedFeat[1]);
	}
	return 0;
}
