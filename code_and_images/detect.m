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
    pause;
    fprintf('got preds for image %d/%d\n', i,nImages);
end


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

load('my_svm.mat')



for i=1:nImages
    % load and show the image
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    scaledImages = {1, 2, 3, 4; ...
        im, imresize(im, 1/2),imresize(im, 1/4),imresize(im, 1/8)};

    for j=1:width(scaledImages)
        imshow(scaledImages{2,j});
        hold on;
        % generate a grid of features across the entire image. you may want to 
        % try generating features more densely (i.e., not in a grid)
        feats = vl_hog(scaledImages{2,j},cellSize);

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
        bestPredicts = 20;
        if (bestPredicts > length(inds))
            bestPredicts = length(inds);
        end
        inds = inds(1:bestPredicts); % (use a bigger number for better recall)
        
        for n=1:numel(inds)        
            [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
            conf = confs(row,col);            
%             if (conf < 0.7)
%                 continue
%             end
            
            bbox = [ col*cellSize*(2^(j-1)) ...
                     row*cellSize*(2^(j-1)) ...
                    (col+cellSize-1)*cellSize*(2^(j-1)) ...
                    (row+cellSize-1)*cellSize*(2^(j-1))];
            
            image_name = {imageList(i).name};
            % save         
            bboxes = [bboxes; bbox];
            confidences = [confidences; conf];
            image_names = [image_names; image_name];
        end

    end
end

%non-max suprression

[sortedConfs, oIndx] = sort(confidences, 'descend');
finalBBoxes = zeros(0,4);

for i=1:height(sortedConfs)
    curBB = bboxes(oIndx(i),:);
    curImName = image_names(oIndx(i),1);
    
    saveToggle = true;
    for j=1:height(finalBBoxes)
        if (curImName == )
            
            pBox = finalBBoxes(j,:);
            bi=[max(curBB(1),pBox(1)) ; max(curBB(2),pBox(2)) ...
                ; min(curBB(3),pBox(3)) ; min(curBB(4),pBox(4))];
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            ua=(curBB(3)-curBB(1)+1)*(curBB(4)-curBB(2)+1)+...
               (pBox(3)-pBox(1)+1)*(pBox(4)-pBox(2)+1)-...
               iw*ih;
            overlap=iw*ih/ua;
            if (overlap > thres)
                saveToggle = false;
                break;
            end
        end
    end
    if (saveToggle)
        finalBBoxes = [finalBBoxes; curBB];
    end
end


% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);