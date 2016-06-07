% Matlab script to manually annotate an image.
% img should be the image to be annotated.

% After annotating all cells, ctrl-C to close image window.

imshow(imread(img));
hold on;

for k = 1:100
    % Get input on the 4 bounding points of the cell
    % (the points can be provided in any order)
    [x,y] = ginput(4);

    % Add the new points to the X and Y matries, and save to disk
    X = round([X; x']);
    Y = round([Y; y']);
    csvwrite('X.csv', X)
    csvwrite('Y.csv', Y)

    % Plot the new points so we can keep track of progress.
    for i = 1:4
        j = mod(i, 4) + 1;
        plot([x(i), x(j)], [y(i), y(j)], 'r', 'Linewidth', 3)
    end
end
