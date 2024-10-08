function lesion_map = artifical_lesion(atlas_mat, feasible_regions, lesion_portion)
% function lesion_map = artifical_lesion(atlas_mat, feasible_regions)
%
% This function creates an artifical lesion in the brain, whose boarder is
% defined by the atlas. This doesn't matter much since the final ouput is a
% binary mask (so, if the individual brain has nothing in a voxel, then
% this mask would not function, even if it has a 1 at that voxel). But of
% course it is surely better to inlcude a subj-spec white matter mask.
%
% 
% 
%
% Input:
%   atlas_mat: The label atlas file, as a 3D matrix. Each value of this
%              matrix indicate the region label of that particular voxel.
%   feasible_regions: A list of integers indicating the brain regions in
%                     which you want to fake lesion. The range should be
%                     the same as the atlas label.
%   lesion_portion: a number in (0,1). The determines the lesion size:
%                   (# of lesion voxels) = lesion_portion * (# of feasible
%                   voxels)
%                     
% Output: 
%   lesion_map: A binary matrix of the same size as atlas_mat. Value 1
%               indicates the voxel being in the lesion.
%
% Example:
%   atlas_mat = niftiread('AAL3v1.nii');
%   lesion_map = artifical_lesion(atlas_mat, 1:60);
%
% Addition notes: if you want to write the result as a NIFTI file, use:
%   lesion_map = uint8(lesion_map);
%   info = niftiinfo("AAL3v1.nii");
%   niftiwrite(lesion_map, 'lsmap_art.nii', info);
%
% Sanity CHECKED by overlaying the lesion on the atlas in FSLeyes.
%
% Modification history:
%   10/07/2023, added, by Zijian
%   10/11/2023, added the option of WM mask, by Zijian
%               added the lesion_portion input, by Zijian


% Extract a binary matrix indicating the feasible voxels
feasible_voxels = ismember(atlas_mat,feasible_regions);
num_feasible_voxels = sum(feasible_voxels,"all");

% Obtain the 3D coordinates of the location of the feasible voxels
linear_indices = find(feasible_voxels);
fvoxel_coord = zeros(num_feasible_voxels,3);
[fvoxel_coord(:,1), fvoxel_coord(:,2), fvoxel_coord(:,3)] = ind2sub(size(atlas_mat), linear_indices);

desired_lesion_size = round(lesion_portion * num_feasible_voxels);

% Randomly select a seed voxel from the feasible voxels
seed_voxel = fvoxel_coord(randi(num_feasible_voxels), :);

% Initialize the lesion region with the seed voxel
lesion_region = seed_voxel;

% Keep growing the lesion region until it reaches the desired size
while size(lesion_region, 1) < desired_lesion_size
    % Find neighboring voxels of the current lesion region
    neighbors = [
        lesion_region + [1, 0, 0];
        lesion_region - [1, 0, 0];
        lesion_region + [0, 1, 0];
        lesion_region - [0, 1, 0];
        lesion_region + [0, 0, 1];
        lesion_region - [0, 0, 1]
    ];

    % Filter neighbors to keep only feasible voxels that are not in the lesion yet
    feasible_neighbors = setdiff(neighbors, lesion_region, 'rows', 'stable');
    feasible_neighbors = feasible_neighbors(all(ismember(feasible_neighbors, fvoxel_coord,"rows"), 2), :);
    
    % If there are no feasible neighbors left, break the loop
    if isempty(feasible_neighbors)
        break;
    end
    
    % Randomly select a feasible neighbor to add to the lesion
    new_voxel_index = randi(size(feasible_neighbors, 1));
    new_voxel = feasible_neighbors(new_voxel_index, :);

    % Add the new voxel to the lesion region
    lesion_region = [lesion_region; new_voxel];

end


% Create a binary matrix representing the lesion region
lesion_map = zeros(size(atlas_mat));
for i = 1:size(lesion_region, 1)
    x = lesion_region(i, 1);
    y = lesion_region(i, 2);
    z = lesion_region(i, 3);
    lesion_map(x, y, z) = 1;
end



% If you are introducing a WM mask, uncomment the following line:
% lesion_map = lesion_map.*WM_mask;


%% !!NEW: dilate the lesion so that there will be no holes inside.

se = offsetstrel('ball',5,5);
lesion_map = imdilate(lesion_map,se);
lesion_map(lesion_map<=5)=0;
lesion_map(lesion_map>5)=1;

% the dilated image may go beyond the WM boundary, but as we mentioned
% above, this doesn't matter much.


end



