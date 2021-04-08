%run('../vlfeat-0.9.20/toolbox/vl_setup')
imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

cellSize = 6;
dim = 36;
thres = 0.3;

load('my_svm.mat')

for i=1:nImages
    cla reset;
    % load and show the image
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
     imshow(im);
     hold on;
    
    % generate a grid of features across the entire image. you may want to 
    % try generating features more densely (i.e., not in a grid)
    feats = vl_hog(im,cellSize);
    
    % concatenate the features into 6x6 bins, and classify them (as if they
    % represent 36x36-pixel faces)
    [rows,cols,~] = size(feats);    
    confs = zeros(rows,cols);
    for r=1:rows-5
        for c=1:cols-5

        % create feature vector for the current window and classify it using the SVM model, 
        featureV = feats(r:r+5,c:c+5,:);
        featureV = reshape(featureV, [1,cellSize*cellSize*31]);
        % take dot product between feature vector and w and add b,
        classified = featureV*w + b;
        % store the result in the matrix of confidence scores confs(r,c)
        confs(r,c) = classified;

        end
    end
       
    % get the most confident predictions 
    [~,inds] = sort(confs(:),'descend');
    inds = inds(1:20); % (use a bigger number for better recall)
    boxesAdded = 1;
    for n=1:numel(inds)        
        [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
        
        bbox = [ col*cellSize ...
                 row*cellSize ...
                (col+cellSize-1)*cellSize ...
                (row+cellSize-1)*cellSize];
        conf = confs(row,col);
        
        if (conf <= 0.7)
            continue;
        end
        
        image_name = {imageList(i).name};
        
        saveToggle = true;
        pCount = 1;
        %greedy non-max suppression
        for pInd=height(bboxes):-1:height(bboxes)-boxesAdded+2
            pBox = bboxes(pInd,:);
            
            bi=[max(bbox(1),pBox(1)) ; max(bbox(2),pBox(2)) ...
                ; min(bbox(3),pBox(3)) ; min(bbox(4),pBox(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            ua=(bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1)+...
               (pBox(3)-pBox(1)+1)*(pBox(4)-pBox(2)+1)-...
               iw*ih;
            overlap=iw*ih/ua;
            disp(overlap)
            if (overlap > thres)
                saveToggle = false;
                [pRow,pCol] = ind2sub([size(feats,1) size(feats,2)],inds(n-pCount));
                pConf = confs(pRow,pCol);
                if (pConf < conf)
                    bboxes(pInd,:) = bbox;
                end
                break;
            end
            pCount = pCount+1;
        end
        if (saveToggle)
            boxesAdded = boxesAdded+1;
            % plot
            plot_rectangle = [bbox(1), bbox(2); ...
                bbox(1), bbox(4); ...
                bbox(3), bbox(4); ...
                bbox(3), bbox(2); ...
                bbox(1), bbox(2)];
            plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');

            % save         
            bboxes = [bboxes; bbox];
            confidences = [confidences; conf];
            image_names = [image_names; image_name];
        end
    end
    %
    fprintf('got preds for image %d/%d\n', i,nImages);
    pause;
end

%%
% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);

%% Face Detection w/ multi-scale analysis & non-greedy non-max suppression

imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

cellSize = 6;
dim = 36;
thres = 0.5;
scales = [1,1/2,1/4,1/8,1/16,1/32,1/64,1/128];
load('my_svm.mat')

for i=1:nImages
    % load and show the image
    cla reset;
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    
    im_bboxes = zeros(0,4);
    im_confidences = zeros(0,1);
    im_image_names = cell(0,1);
    
    for j=1:width(scales)
        imResized = imresize(im,scales(j));
        imshow(imResized);
        hold on;
        % generate a grid of features across the entire image. you may want to 
        % try generating features more densely (i.e., not in a grid)
        feats = vl_hog(imResized,cellSize);

        % concatenate the features into 6x6 bins, and classify them (as if they
        % represent 36x36-pixel faces)
        [rows,cols,~] = size(feats);    
        confs = zeros(rows,cols);
        for r=1:rows-5
            for c=1:cols-5

            % create feature vector for the current window and classify it using the SVM model, 
            featureV = feats(r:r+5,c:c+5,:);
            featureV = reshape(featureV, [1,cellSize*cellSize*31]);
            % take dot product between feature vector and w and add b,
            classified = featureV*w + b;
            % store the result in the matrix of confidence scores confs(r,c)
            confs(r,c) = classified;

            end
        end
        % get the most confident predictions 
        [~,inds] = sort(confs(:),'descend');
        bestPredicts = 40;
        if (bestPredicts > length(inds))
            bestPredicts = length(inds);
        end
        inds = inds(1:bestPredicts); % (use a bigger number for better recall)
        
        for n=1:numel(inds)        
            [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
            conf = confs(row,col);            
            if (conf < 0.90)
                continue
            end
            
            bbox = [ col*cellSize*(scales(j)^-1) ...
                     row*cellSize*(scales(j)^-1) ...
                    (col+cellSize-1)*cellSize*(scales(j)^-1) ...
                    (row+cellSize-1)*cellSize*(scales(j)^-1)];
            
            image_name = {imageList(i).name};
            % save         
            im_bboxes = [im_bboxes; bbox];
            im_confidences = [im_confidences; conf];
            im_image_names = [im_image_names; image_name];
        end

    end
    
    [sortedConfs, oIndx] = sort(im_confidences, 'descend');
    finalBBoxes = zeros(0,5);
    
    for indx=1:height(sortedConfs)
        curBB = zeros(0,5);
        curBB(1:4) = im_bboxes(oIndx(indx),:);
        curBB(5) = oIndx(indx);
        
        saveToggle = true;
        for j=1:height(finalBBoxes)
            pBox = finalBBoxes(j,:);
            bi=[max(curBB(1),pBox(1)) ; max(curBB(2),pBox(2)) ...
                ; min(curBB(3),pBox(3)) ; min(curBB(4),pBox(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            ua=(curBB(3)-curBB(1)+1)*(curBB(4)-curBB(2)+1)+...
               (pBox(3)-pBox(1)+1)*(pBox(4)-pBox(2)+1)-...
               iw*ih;
            overlap=iw*ih/ua;
            if (overlap > 0.5)
                saveToggle = false;
                break;
            elseif (iw*ih == (curBB(3)-curBB(1)+1) * (curBB(4)-curBB(2)+1) || iw*ih == (pBox(3)-pBox(1)+1) * (pBox(4)-pBox(2)+1))
                saveToggle = false;
                break;
            end
        end

        if (saveToggle)
            finalBBoxes = [finalBBoxes; curBB];
        end
    end
    
    finalConfs = zeros([height(finalBBoxes),1]);
    finalImNames = strings([height(finalBBoxes),1])+image_name;

    for indx=1:height(finalBBoxes)
        finalConfs(indx,1) = im_confidences(finalBBoxes(indx,5));
    end
    imshow(im);
    for j=1:height(finalBBoxes)
        curBB = finalBBoxes(j,:);
        % plot
        plot_rectangle = [curBB(1), curBB(2); ...
            curBB(1), curBB(4); ...
            curBB(3), curBB(4); ...
            curBB(3), curBB(2); ...
            curBB(1), curBB(2)];
        plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
    end
    fprintf('got preds for image %d/%d\n', i,nImages);    
    
    if (length(finalBBoxes)>0)
        bboxes = [bboxes; finalBBoxes(:,1:4)];
        confidences = [confidences; finalConfs];
        image_names = [image_names; finalImNames];
    end
    %pause;
end
%%
% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);