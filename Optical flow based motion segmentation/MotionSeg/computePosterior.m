 function [bgMaskPosterior, fgMaskPosterior, newMotionPosterior, bgPrior, fgPrior, newMotionPrior] = computePosteriors( likelihood_bg, likelihood_fg, likelihood_newMotion, OF, bgPrior, fgPrior, n, newPrior)

[height, width, ~] = size(bgPrior);

    norm = (bgPrior + sum(fgPrior,3))./(1-newPrior);
    bgPriorNorm  = (bgPrior)./norm;
    bgPriorNorm(isnan(bgPriorNorm)) =0.9*(1-1/(n+2));
    norm = repmat(norm,1,1,n);
    if (n~=0)
        fgPriorNorm = (fgPrior)./norm;
        fgPriorNorm(isnan(fgPriorNorm)) =0.1*(1-1/(n+2))/n;
    else
        fgPriorNorm = fgPrior;
    end
    
    %Smooth the prior
    fLength = 7;
    filter = fspecial('gaussian', fLength, fLength/4);
    bgPrior = imfilter( bgPriorNorm, filter,'replicate');
    
    for i = 1:n
        fgPrior(:,:,i) = imfilter( fgPriorNorm(:,:,i), filter,'replicate');
    end
    
    norm = (bgPrior + sum(fgPrior,3))./(1-newPrior);
    bgPriorNorm  = (bgPrior)./norm;
    norm = repmat(norm,1,1,n);
    if (n~=0)
        fgPriorNorm = (fgPrior)./norm;
    else
        fgPriorNorm = fgPrior;
    end
    
    denominator = ((likelihood_bg.*bgPriorNorm) + sum((likelihood_fg(:,:,1:n).*fgPriorNorm(:,:,1:n)),3)+likelihood_newMotion.*newPrior);
       
    %Posteriors for each motion component and BG
    bgMaskPosterior = (likelihood_bg.*bgPriorNorm)...
            ./denominator;
        
    fgMaskPosterior = (likelihood_fg(:,:,1:n).*fgPriorNorm(:,:,1:n))./repmat(denominator,1,1,n);
    
    newMotionPosterior = (likelihood_newMotion.*newPrior)...
            ./denominator;


    %Priors for next fame
    bgPrior = Posterior2Prior(bgMaskPosterior, OF);
    
    fgPrior = zeros(height, width, n);
    for i = 1:n
        fgPrior(:,:,i) = Posterior2Prior(fgMaskPosterior(:,:,i), OF);
    end
    
    newMotionPrior = Posterior2Prior(newMotionPosterior, OF);
    
end

        
