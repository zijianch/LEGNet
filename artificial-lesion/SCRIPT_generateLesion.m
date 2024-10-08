% The Arterial Atlas is used to determine where the stroke lesion can grow.
%
% Description: We present an atlas of brain arterial territories based on 
% lesion distributions in 1,298 acute stroke patients. The atlas covers 
% supra- and infra-tentorial regions and contains hierarchical segmentation 
% levels created by a fusion of vascular and classical anatomical criteria.
%
%
% URL: https://www.nitrc.org/projects/arterialatlas
% 
% Liu, CF., Hsu, J., Xu, X. et al. Digital 3D Brain MRI Arterial 
% Territories Atlas. Sci Data 10, 74 (2023). https://doi.org/10.1038/s41597-022-01923-0
%
% Look-up table:
%
% +---------------------------------------------------+------------------+------------+------------------+--------------+-----------------+-----------------+
% | Arterial Territories                              | Level1 Intensity | Level1     | Level2 Intensity | Level2       | Level2 Bilateral| Major           |
% |                                                   |                  | Labels     |                  | Labels       |                 | Territories     |
% +---------------------------------------------------+------------------+------------+------------------+--------------+-----------------+-----------------+
% | anterior cerebral artery left                     | 1                | ACAL       | 1                | ACAL         | ACA             | Anterior        |
% | anterior cerebral artery right                    | 2                | ACAR       | 2                | ACAR         | ACA             | Anterior        |
% | medial lenticulostriate left                      | 3                | MLSL       | 1                | ACAL         | ACA             | Anterior        |
% | medial lenticulostriate right                     | 4                | MLSR       | 2                | ACAR         | ACA             | Anterior        |
% | lateral lenticulostriate left                     | 5                | LLSL       | 3                | MCAL         | MCA             | Anterior        |
% | lateral lenticulostriate right                    | 6                | LLSR       | 4                | MCAR         | MCA             | Anterior        |
% | frontal pars of middle cerebral artery left       | 7                | MCAFL      | 3                | MCAL         | MCA             | Anterior        |
% | frontal pars of middle cerebral artery right      | 8                | MCAFR      | 4                | MCAR         | MCA             | Anterior        |
% | parietal pars of middle cerebral artery left      | 9                | MCAPL      | 3                | MCAL         | MCA             | Anterior        |
% | parietal pars of middle cerebral artery right     | 10               | MCAPR      | 4                | MCAR         | MCA             | Anterior        |
% | temporal pars of middle cerebral artery left      | 11               | MCATL      | 3                | MCAL         | MCA             | Anterior        |
% | temporal pars of middle cerebral artery right     | 12               | MCATR      | 4                | MCAR         | MCA             | Anterior        |
% | occipital pars of middle cerebral artery left     | 13               | MCAOL      | 3                | MCAL         | MCA             | Anterior        |
% | occipital pars of middle cerebral artery right    | 14               | MCAOR      | 4                | MCAR         | MCA             | Anterior        |
% | insular pars of middle cerebral artery left       | 15               | MCAIL      | 3                | MCAL         | MCA             | Anterior        |
% | insular pars of middle cerebral artery right      | 16               | MCAIR      | 4                | MCAR         | MCA             | Anterior        |
% | temporal pars of posterior cerebral artery left   | 17               | PCATL      | 5                | PCAR         | PCA             | Posterior       |
% | temporal pars of posterior cerebral artery right  | 18               | PCATR      | 6                | PCAL         | PCA             | Posterior       |
% | occipital pars of posterior cerebral artery left  | 19               | PCAOL      | 5                | PCAL         | PCA             | Posterior       |
% | occipital pars of posterior cerebral artery right | 20               | PCAOR      | 6                | PCAR         | PCA             | Posterior       |
% | posterior choroidal and thalamoperfurators left   | 21               | PCTPL      | 5                | PCAL         | PCA             | Posterior       |
% | posterior choroidal and thalamoperfurators right  | 22               | PCTPR      | 6                | PCAR         | PCA             | Posterior       |
% | anterior choroidal and thalamoperfurators left    | 23               | ACTPL      | 5                | PCAL         | PCA             | Posterior       |
% | anterior choroidal and thalamoperfurators right   | 24               | ACTPR      | 6                | PCAR         | PCA             | Posterior       |
% | basilar left                                      | 25               | BL         | 7                | VBL          | VB              | Posterior       |
% | basilar right                                     | 26               | BR         | 8                | VBR          | VB              | Posterior       |
% | superior cerebellar left                          | 27               | SCL        | 7                | VBL          | VB              | Posterior       |
% | superior cerebellar right                         | 28               | SCR        | 8                | VBR          | VB              | Posterior       |
% | inferior cerebellar left                          | 29               | ICL        | 7                | VBL          | VB              | Posterior       |
% | inferior cerebellar right                         | 30               | ICR        | 8                | VBR          | VB              | Posterior       |
% | lateral ventricle left                            | 31               | LVL        | 9                | LVL          | LV              | Ventricle       |
% | lateral ventricle right                           | 32               | LVR        | 10               | LVR          | LV              | Ventricle       |
% +---------------------------------------------------+------------------+------------+------------------+--------------+----------------+-----------------+
% 
%
% Note: this is 182x218x182 int16 format (so it's 1mm resolution)


num = 100; % number of total lesion masks to be generated

atlas_mat = niftiread('/project/ArterialAtlas_level2.nii');
atlas_nii_info_A = niftiinfo('/project/ArterialAtlas_level2.nii');
atlas_nii_info_S = niftiinfo('/project/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');


for i = 1:num
    
    lesion_portion =  0.01 + (0.15 - 0.01) * rand; % random portio 0.02-0.2
    feasible_regions = [1,3, 5, 7];

    target_region = datasample(feasible_regions, 1); % excluding ventricles

    fprintf('target region = %d, lesion portion = %f\n', target_region, lesion_portion);

    lesion_map = artifical_lesion(atlas_mat, target_region, lesion_portion);
    
    output_filename = sprintf('/project/artificial-lesion-maps/lesionMap_%d.nii', i);
    niftiwrite(int16(lesion_map), output_filename, atlas_nii_info_A);

    if any(size(lesion_map) ~= [91 109 91])
        lesion_map = imresize3(lesion_map,[91 109 91]);
        lesion_map(lesion_map>=0.5) = 1;
        lesion_map(lesion_map<0.5) = 0;
    end
    
    output_filename = sprintf('/project/artificial-lesion-maps/lesionMap_%d_Sch.nii', i);
    niftiwrite(single(lesion_map), output_filename, atlas_nii_info_S);
    
end

