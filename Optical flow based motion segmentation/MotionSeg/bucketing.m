function [ patch ] = bucketing( segments, bucket_matrix )
%BUCKETING Summary of this function goes here
%   Detailed explanation goes here

random_bucket = randperm(4,3);

% select three patches of the four corner buckets
patch = 0;
for bucket_num = 1:3
    
    C = unique(bucket_matrix(:,:,random_bucket(bucket_num)));
    r = randperm(length(C),1);
    r = C(r);
    
    patch = patch + ismember(segments, r);
    
end

% select five patches randomly
mask = uint32((patch.*2 - 1).*(-1));
segments = segments.*mask;
C = unique(segments(segments>0));

r = randperm(length(C),7);
r = C(r);

patch = patch + ismember(segments, r);

end

