# --------------------------------------------------
# PW KERNEL COMPUTATION FOR BIOLOGICAL SEQUENCES
#
# @author: dimiboeckaerts
# --------------------------------------------------

# GENERAL INFO
# --------------------------------------------------
"""
In this script, BioSequences and BioAlignments packages are used to compute a
kernel matrix from protein sequences based on pairwise alignment scores. This is
done using three different functions:
- 'file_to_list' converts a FASTA file to a list over which we can loop.
- 'calculate_pwkernel_matrix' computes the kernel matrix for a given FASTA file,
    using 'file_to_list' and 'BioAlignments'.
"""

# IMPORT LIBRARIES
# --------------------------------------------------
# import Pkg
# Pkg.add("BioSequences")
# Pkg.add("BioAlignments")
using BioSequences
using BioAlignments
using DelimitedFiles
using LinearAlgebra
using ProgressMeter


# FUNCTIONS FOR ALIGNMENT SCORE COMPUTATION
# --------------------------------------------------
# convert file to a list
function file_to_array(file)
    """
    Function that reads a FASTA file and puts its sequences in an array.
    """
    sequences = []
    reader = FASTA.Reader(open(file, "r"))
    for record in reader
        seq = FASTA.sequence(record)
        push!(sequences, seq)
    end
    return sequences
end

# calculate alingment and its identity/match score
function calculate_perc_identity(sequence1, sequence2)
    """
    This function calculates the percentage of matches between two aligned protein sequences.
    """
    scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1)
    res = pairalign(LocalAlignment(), sequence1, sequence2, scoremodel);
    aln = alignment(res)

    return count_matches(aln) / min(length(sequence1), length(sequence2))
end

# construct alignment score matrix
function calculate_score_matrix(file)
    """
    Function that constructs a kernel matrix based on pairwise sequence alignment.

    Input: a file handle to a FASTA file
    Output: a kernel matrix
    """
    # for simple alignment: AffineGapScoreModel(match=1, mismatch=0, gap_open=0, gap_extend=0)
    # for BLOSUM alignment: AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1)
    # by adding the transposed matrix, we add the diagonal twice... this should be corrected.

    sequence_list = file_to_array(file)
    kernel_matrix = zeros(length(sequence_list), length(sequence_list))
    p = Progress(Int64(round((length(sequence_list)^2)/2, digits=0)))

    for i in 1:length(sequence_list), j in i:length(sequence_list)
        kernel_matrix[i,j] = calculate_perc_identity(sequence_list[i], sequence_list[j])
        next!(p)
    end

    kernel_matrix = kernel_matrix + kernel_matrix'
    kernel_matrix = kernel_matrix - Diagonal(kernel_matrix)/2
    return kernel_matrix
end


# TIMING TEST
# --------------------------------------------------
test_seq1 = aa"MTDIITNVVIGMPSQLFTMARSFKAVANGKIYIGKIDTDPVNPENQIQVYVENEDGSHVPASQPIVINAAGYPVYNGQIVKFVTEQGHSMAVYDAYGSQQFYFQNVLKYDPDQFGPDLIEQLAQSGKYSQDNTKGDAMIGVKQPLPKAVLRTQHDKNKEAISILDFGV"
test_seq2 = aa"MAITKIILQQMVTMDQNSITASKYPKYTVVLSNSISSITAADVTSAIESSKASGPAAKQSEINAKQSELNAKDSENEAEISATSSQQSATQSASSATASANSAKAAKTSETNANNSKNAAKTSETNAASSASSASSFATAAENSARAAKTSETNAGNSAQAADASKTA"

scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1)
res = pairalign(LocalAlignment(), test_seq1, test_seq2, scoremodel);
aln = alignment(res)
sc = score(res)
count_matches(aln)


# COMPUTE KERNEL AND SAVE MATRIX TO FILE
# --------------------------------------------------
seqfile = "/Users/Dimi/Documents/GitHub_Local/BacteriophageHostPrediction/RBP_sequences.fasta"
m = calculate_score_matrix(seqfile)
size(m)
writedlm("/Users/Dimi/Documents/GitHub_Local/BacteriophageHostPrediction/RBP_alignmentscores.txt", m)
matrix = readdlm("/Users/Dimi/Documents/GitHub_Local/BacteriophageHostPrediction/RBP_alignmentcores.txt")


# COMPUTE ALIGNMENT FEATURES BASED ON RANDOM SAMPLES
# --------------------------------------------------
seqfile = "/Users/Dimi/Documents/GitHub_Local/BacteriophageHostPrediction/RBP_sequences887.fasta"
sequence_list = file_to_array(seqfile)
scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1)
sample_list = [633, 229, 404, 434, 415, 442, 848, 421, 245, 375, 748, 741, 249, 361, 492, 525, 523, 800, 255, 390, 261, 452, 406, 288, 305, 237, 394, 446, 485, 459, 501, 209, 396, 186, 184, 500, 554, 420, 511, 339, 479, 530, 317, 296, 367, 469, 215, 508, 381, 542, 343, 397, 425, 219, 276, 541, 467, 218, 497, 347, 388, 238, 489, 559, 676, 400, 869, 437, 201, 408, 365, 564, 353, 412, 308, 372, 555, 351, 494, 244, 522, 486, 312, 331, 730, 714, 876, 3, 638, 757, 54, 602, 13, 88, 767, 594, 154, 102, 75, 814, 771, 590, 159, 752, 23, 760, 654, 679, 701, 675, 782, 86, 703, 871, 828, 664, 605, 844, 665, 671, 711, 687, 612, 716, 731, 685, 570, 587, 585, 574, 569, 576, 575, 579, 577, 573, 581, 571, 584, 588, 583, 580, 586, 833, 582, 572, 578]
feature_matrix = zeros(length(sequence_list), length(sample_list))
p = Progress(Int64(round((length(sequence_list)*length(sample_list)), digits=0)))


for i in 1:length(sequence_list), j in 1:length(sample_list)
    res = pairalign(LocalAlignment(), sequence_list[i], sequence_list[sample_list[j]], scoremodel)
    feature_matrix[i,j] = score(res)
    next!(p)
end

writedlm("/Users/Dimi/Documents/GitHub_Local/BacteriophageHostPrediction/RBP_alignment_features.txt", feature_matrix)
