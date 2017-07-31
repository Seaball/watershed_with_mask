iNumOfSpecs#include <vector>
#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

struct WSNode
{
    int next;
    int mask_ofs;
    int img_ofs;
};

// Queue for WSNodes
struct WSQueue
{
    WSQueue() { first = last = 0; }
    int first, last;
};


static int allocWSNodes(std::vector<WSNode>& storage)
{
    int sz = (int)storage.size();
    int newsz = MAX(128, sz * 3 / 2);

    storage.resize(newsz);
    if (sz == 0)
    {
        storage[0].next = 0;
        sz = 1;
    }
    for (int i = sz; i < newsz - 1; i++)
        storage[i].next = i + 1;
    storage[newsz - 1].next = 0;
    return sz;
}

//the modified version of watershed algorithm from OpenCV
void watershedEx(InputArray _src, InputOutputArray _markers)
{
    // Labels for pixels
    const int IN_QUEUE = -2; // Pixel visited
    // possible bit values = 2^8
    const int NQ = 256;

    Mat src = _src.getMat(), dst = _markers.getMat();
    Size size = src.size();
    int channel = src.channels();
    // Vector of every created node
    std::vector<WSNode> storage;
    int free_node = 0, node;
    // Priority queue of queues of nodes
    // from high priority (0) to low priority (255)
    WSQueue q[NQ];
    // Non-empty queue with highest priority
    int active_queue;
    int i, j;
    // Color differences
    int db, dg, dr;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

    // Create a new node with offsets mofs and iofs in queue idx
#define ws_push(idx,mofs,iofs)          \
        {                                       \
    if (!free_node)                    \
    free_node = allocWSNodes(storage); \
    node = free_node;                   \
    free_node = storage[free_node].next; \
    storage[node].next = 0;             \
    storage[node].mask_ofs = mofs;      \
    storage[node].img_ofs = iofs;       \
    if (q[idx].last)                   \
    storage[q[idx].last].next = node; \
    else                                \
  q[idx].first = node;            \
  q[idx].last = node;                 \
        }

    // Get next node from queue idx
#define ws_pop(idx,mofs,iofs)           \
        {                                       \
    node = q[idx].first;                \
    q[idx].first = storage[node].next;  \
    if (!storage[node].next)           \
    q[idx].last = 0;                \
    storage[node].next = free_node;     \
    free_node = node;                   \
    mofs = storage[node].mask_ofs;      \
    iofs = storage[node].img_ofs;       \
        }

    // Get highest absolute channel difference in diff
#define c_diff(ptr1,ptr2,diff)           \
        {                                        \
    db = std::abs((ptr1)[0] - (ptr2)[0]); \
    dg = std::abs((ptr1)[1] - (ptr2)[1]); \
    dr = std::abs((ptr1)[2] - (ptr2)[2]); \
    diff = ws_max(db, dg);                \
    diff = ws_max(diff, dr);              \
    assert(0 <= diff && diff <= 255);  \
        }

    //get absolute difference in diff
#define c_gray_diff(ptr1,ptr2,diff)		\
        {									\
    diff = std::abs((ptr1)[0] - (ptr2)[0]);	\
    assert(0 <= diff&&diff <= 255);		\
        }

    CV_Assert(src.type() == CV_8UC3 || src.type() == CV_8UC1&& dst.type() == CV_32SC1);
    CV_Assert(src.size() == dst.size());

    // Current pixel in input image
    const uchar* img = src.ptr();
    // Step size to next row in input image
    int istep = int(src.step / sizeof(img[0]));

    // Current pixel in mask image
    int* mask = dst.ptr<int>();
    // Step size to next row in mask image
    int mstep = int(dst.step / sizeof(mask[0]));

    for (i = 0; i < 256; i++)
        subs_tab[i] = 0;
    for (i = 256; i <= 512; i++)
        subs_tab[i] = i - 256;

    //for (j = 0; j < size.width; j++)
    //mask[j] = mask[j + mstep*(size.height - 1)] = 0;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for (i = 1; i < size.height - 1; i++){
        img += istep; mask += mstep;
        mask[0] = mask[size.width - 1] = 0; // boundary pixels

        for (j = 1; j < size.width - 1; j++){
            int* m = mask + j;
            if (m[0] < 0)
                m[0] = 0;
            if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
            {
                // Find smallest difference to adjacent markers
                const uchar* ptr = img + j * channel;
                int idx = 256, t;
                if (m[-1] > 0){
                    if (channel == 3){
                        c_diff(ptr, ptr - channel, idx);
                    }
                    else{
                        c_gray_diff(ptr, ptr - channel, idx);
                    }
                }
                if (m[1] > 0){
                    if (channel == 3){
                        c_diff(ptr, ptr + channel, t);
                    }
                    else{
                        c_gray_diff(ptr, ptr + channel, t);
                    }
                    idx = ws_min(idx, t);
                }
                if (m[-mstep] > 0){
                    if (channel == 3){
                        c_diff(ptr, ptr - istep, t);
                    }
                    else{
                        c_gray_diff(ptr, ptr - istep, t);
                    }
                    idx = ws_min(idx, t);
                }
                if (m[mstep] > 0){
                    if (channel == 3){
                        c_diff(ptr, ptr + istep, t);
                    }
                    else{
                        c_gray_diff(ptr, ptr + istep, t);
                    }
                    idx = ws_min(idx, t);
                }

                // Add to according queue
                assert(0 <= idx && idx <= 255);
                ws_push(idx, i*mstep + j, i*istep + j * channel);
                m[0] = IN_QUEUE;//initial unvisited
            }
        }
    }
    // find the first non-empty queue
    for (i = 0; i < NQ; i++)
        if (q[i].first)
            break;

    // if there is no markers, exit immediately
    if (i == NQ)
        return;

    active_queue = i;//first non-empty priority queue
    img = src.ptr();
    mask = dst.ptr<int>();

    // recursively fill the basins
    for (;;)
    {
        int mofs, iofs;
        int lab = 0, t;
        int* m;
        const uchar* ptr;

        // Get non-empty queue with highest priority
        // Exit condition: empty priority queue
        if (q[active_queue].first == 0)
        {
            for (i = active_queue + 1; i < NQ; i++)
                if (q[i].first)
                    break;
            if (i == NQ)
            {
                std::vector<WSNode>().swap(storage); 
                break;
            }
            active_queue = i;
        }

        // Get next node
        ws_pop(active_queue, mofs, iofs);
        int top = 1, bottom = 1, left = 1, right = 1;
        if (0 <= mofs&&mofs < mstep)//pixel on the top
            top = 0;
        if ((mofs%mstep) == 0)//pixel in the left column
            left = 0;
        if ((mofs + 1) % mstep == 0)//pixel in the right column
            right = 0;
        if (mstep*(size.height - 1) <= mofs && mofs < mstep*size.height)//pixel on the bottom
            bottom = 0;

        // Calculate pointer to current pixel in input and marker image
        m = mask + mofs;
        ptr = img + iofs;
        int diff, temp;
        // Check surrounding pixels for labels to determine label for current pixel
        if (left){//the left point can be visited
            t = m[-1];
            if (t > 0){
                lab = t;
                if (channel == 3){
                    c_diff(ptr, ptr - channel, diff);
                }
                else{
                    c_gray_diff(ptr, ptr - channel, diff);
                }
            }
        }
        if (right){// Right point can be visited
            t = m[1];
            if (t > 0){
                if (lab == 0){//and this point didn't be labeled before
                    lab = t;
                    if (channel == 3){
                        c_diff(ptr, ptr + channel, diff);
                    }
                    else{
                        c_gray_diff(ptr, ptr + channel, diff);
                    }
                }
                else if (t != lab){
                    if (channel == 3){
                        c_diff(ptr, ptr + channel, temp);
                    }
                    else{
                        c_gray_diff(ptr, ptr + channel, temp);
                    }
                    diff = ws_min(diff, temp);
                    if (diff == temp)
                        lab = t;
                }
            }
        }
        if (top){
            t = m[-mstep]; // Top
            if (t > 0){
                if (lab == 0){//and this point didn't be labeled before
                    lab = t;
                    if (channel == 3){
                        c_diff(ptr, ptr - istep, diff);
                    }
                    else{
                        c_gray_diff(ptr, ptr - istep, diff);
                    }
                }
                else if (t != lab){
                    if (channel == 3){
                        c_diff(ptr, ptr - istep, temp);
                    }
                    else{
                        c_gray_diff(ptr, ptr - istep, temp);
                    }
                    diff = ws_min(diff, temp);
                    if (diff == temp)
                        lab = t;
                }
            }
        }
        if (bottom){
            t = m[mstep]; // Bottom
            if (t > 0){
                if (lab == 0){
                    lab = t;
                }
                else if (t != lab){
                    if (channel == 3){
                        c_diff(ptr, ptr + istep, temp);
                    }
                    else{
                        c_gray_diff(ptr, ptr + istep, temp);
                    }
                    diff = ws_min(diff, temp);
                    if (diff == temp)
                        lab = t;
                }
            }
        }
        // Set label to current pixel in marker image
        assert(lab != 0);//lab must be labeled with a nonzero number
        m[0] = lab;

        // Add adjacent, unlabeled pixels to corresponding queue
        if (left){
            if (m[-1] == 0)//left pixel with marker 0
            {
                if (channel == 3){
                    c_diff(ptr, ptr - channel, t);
                }
                else{
                    c_gray_diff(ptr, ptr - channel, t);
                }
                ws_push(t, mofs - 1, iofs - channel);
                active_queue = ws_min(active_queue, t);
                m[-1] = IN_QUEUE;
            }
        }

        if (right)
        {
            if (m[1] == 0)//right pixel with marker 0
            {
                if (channel == 3){
                    c_diff(ptr, ptr + channel, t);
                }
                else{
                    c_gray_diff(ptr, ptr + channel, t);
                }
                ws_push(t, mofs + 1, iofs + channel);
                active_queue = ws_min(active_queue, t);
                m[1] = IN_QUEUE;
            }
        }

        if (top)
        {
            if (m[-mstep] == 0)//top pixel with marker 0
            {
                if (channel == 3){
                    c_diff(ptr, ptr - istep, t);
                }
                else{
                    c_gray_diff(ptr, ptr - istep, t);
                }
                ws_push(t, mofs - mstep, iofs - istep);
                active_queue = ws_min(active_queue, t);
                m[-mstep] = IN_QUEUE;
            }
        }

        if (bottom){
            if (m[mstep] == 0)//down pixel with marker 0
            {
                if (channel == 3){
                    c_diff(ptr, ptr + istep, t);
                }
                else{
                    c_gray_diff(ptr, ptr + istep, t);
                }
                ws_push(t, mofs + mstep, iofs + istep);
                active_queue = ws_min(active_queue, t);
                m[mstep] = IN_QUEUE;
            }
        }
    }
}

//the original version of watershed algorithm in OpenCV
void watershedOrg(cv::InputArray _src, cv::InputOutputArray _markers)
{
    // Labels for pixels
    const int IN_QUEUE = -2; // Pixel visited
    const int WSHED = -1; // Pixel belongs to watershed

    // possible bit values = 2^8
    const int NQ = 256;

    cv::Mat src = _src.getMat(), dst = _markers.getMat();
    cv::Size size = src.size();

    // Vector of every created node
    std::vector<WSNode> storage;
    int free_node = 0, node;
    // Priority queue of queues of nodes
    // from high priority (0) to low priority (255)
    WSQueue q[NQ];
    // Non-empty queue with highest priority
    int active_queue;
    int i, j;
    // Color differences
    int db, dg, dr;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

    // Create a new node with offsets mofs and iofs in queue idx
#define ws_push(idx,mofs,iofs)          \
    {                                       \
    if (!free_node)                    \
    free_node = allocWSNodes(storage); \
    node = free_node;                   \
    free_node = storage[free_node].next; \
    storage[node].next = 0;             \
    storage[node].mask_ofs = mofs;      \
    storage[node].img_ofs = iofs;       \
    if (q[idx].last)                   \
    storage[q[idx].last].next = node; \
  else                                \
  q[idx].first = node;            \
  q[idx].last = node;                 \
    }

    // Get next node from queue idx
#define ws_pop(idx,mofs,iofs)           \
    {                                       \
    node = q[idx].first;                \
    q[idx].first = storage[node].next;  \
    if (!storage[node].next)           \
    q[idx].last = 0;                \
    storage[node].next = free_node;     \
    free_node = node;                   \
    mofs = storage[node].mask_ofs;      \
    iofs = storage[node].img_ofs;       \
    }

    // Get highest absolute channel difference in diff
#define c_diff(ptr1,ptr2,diff)           \
    {                                        \
    db = std::abs((ptr1)[0] - (ptr2)[0]); \
    dg = std::abs((ptr1)[1] - (ptr2)[1]); \
    dr = std::abs((ptr1)[2] - (ptr2)[2]); \
    diff = ws_max(db, dg);                \
    diff = ws_max(diff, dr);              \
    assert(0 <= diff && diff <= 255);  \
    }

    CV_Assert(src.type() == CV_8UC3 && dst.type() == CV_32SC1);
    CV_Assert(src.size() == dst.size());

    // Current pixel in input image
    const uchar* img = src.ptr();
    // Step size to next row in input image
    int istep = int(src.step / sizeof(img[0]));

    // Current pixel in mask image
    int* mask = dst.ptr<int>();
    // Step size to next row in mask image
    int mstep = int(dst.step / sizeof(mask[0]));

    for (i = 0; i < 256; i++)
        subs_tab[i] = 0;
    for (i = 256; i <= 512; i++)
        subs_tab[i] = i - 256;

    // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
    for (j = 0; j < size.width; j++)
        mask[j] = mask[j + mstep*(size.height - 1)] = WSHED;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for (i = 1; i < size.height - 1; i++)
    {
        img += istep; mask += mstep;
        mask[0] = mask[size.width - 1] = WSHED; // boundary pixels

        for (j = 1; j < size.width - 1; j++)
        {
            int* m = mask + j;
            if (m[0] < 0)
                m[0] = 0;
            if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
            {
                // Find smallest difference to adjacent markers
                const uchar* ptr = img + j * 3;
                int idx = 256, t;
                if (m[-1] > 0)
                    c_diff(ptr, ptr - 3, idx);
                if (m[1] > 0)
                {
                    c_diff(ptr, ptr + 3, t);
                    idx = ws_min(idx, t);
                }
                if (m[-mstep] > 0)
                {
                    c_diff(ptr, ptr - istep, t);
                    idx = ws_min(idx, t);
                }
                if (m[mstep] > 0)
                {
                    c_diff(ptr, ptr + istep, t);
                    idx = ws_min(idx, t);
                }
                // Add to according queue
                assert(0 <= idx && idx <= 255);
                ws_push(idx, i*mstep + j, i*istep + j * 3);
                m[0] = IN_QUEUE;//initial unvisited
            }
        }
    }

    // find the first non-empty queue
    for (i = 0; i < NQ; i++)
    if (q[i].first)
        break;

    // if there is no markers, exit immediately
    if (i == NQ)
        return;

    active_queue = i;//first non-empty priority queue
    img = src.ptr();
    mask = dst.ptr<int>();

    // recursively fill the basins
    for (;;)
    {
        int mofs, iofs;
        int lab = 0, t;
        int* m;
        const uchar* ptr;

        // Get non-empty queue with highest priority
        // Exit condition: empty priority queue
        if (q[active_queue].first == 0)
        {
            for (i = active_queue + 1; i < NQ; i++)
            if (q[i].first)
                break;
            if (i == NQ){
                break;
            }

            active_queue = i;
        }

        // Get next node
        ws_pop(active_queue, mofs, iofs);

        // Calculate pointer to current pixel in input and marker image
        m = mask + mofs;
        ptr = img + iofs;
        // Check surrounding pixels for labels to determine label for current pixel
        t = m[-1]; // Left
        if (t > 0){
            lab = t;
        }
        t = m[1]; // Right
        if (t > 0){
            if (lab == 0)//and this point didn't be labeled before
                lab = t;
            else if (t != lab)
                lab = WSHED;
        }
        t = m[-mstep]; // Top
        if (t > 0){
            if (lab == 0)//and this point didn't be labeled before
                lab = t;
            else if (t != lab){
                lab = WSHED;
            }
        }
        t = m[mstep]; // Bottom
        if (t > 0){
            if (lab == 0){
                lab = t;
            }
            else if (t != lab){
                lab = WSHED;
            }
        }

        // Set label to current pixel in marker image
        assert(lab != 0);//lab must be labeled with a nonzero number
        m[0] = lab;

        if (lab == WSHED)
            continue;

        // Add adjacent, unlabeled pixels to corresponding queue
        if (m[-1] == 0)//left pixel with marker 0
        {
            c_diff(ptr, ptr - 3, t);
            ws_push(t, mofs - 1, iofs - 3);
            active_queue = ws_min(active_queue, t);
            m[-1] = IN_QUEUE;
        }
        if (m[1] == 0)//right pixel with marker 0
        {
            c_diff(ptr, ptr + 3, t);
            ws_push(t, mofs + 1, iofs + 3);
            active_queue = ws_min(active_queue, t);
            m[1] = IN_QUEUE;
        }
        if (m[-mstep] == 0)//top pixel with marker 0
        {
            c_diff(ptr, ptr - istep, t);
            ws_push(t, mofs - mstep, iofs - istep);
            active_queue = ws_min(active_queue, t);
            m[-mstep] = IN_QUEUE;
        }
        if (m[mstep] == 0)//down pixel with marker 0
        {
            c_diff(ptr, ptr + istep, t);
            ws_push(t, mofs + mstep, iofs + istep);
            active_queue = ws_min(active_queue, t);
            m[mstep] = IN_QUEUE;
        }
    }
}

void main(int argc, char ** argv)
{
    vector<Vec3b> colorTab;
    for (int i = 0; i <= 600; i++){
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    Mat image = imread(argv[1], -1);    //image with 3 channels or 1 channels
    imshow("Source Image", image);
    waitKey(0);

    Mat marks(image.size(), CV_32S);   //make marks
    marks = Scalar::all(0);
    int region_size = atoi(argv[2]);
    int label = 1;
    for (int i = region_size / 2; i < marks.rows; i += region_size) {
        for (int j = region_size / 2; j < marks.cols; j += region_size) {
            marks.at<int>(i, j) = label;
            label++;
        }
    }
#ifdef _MASK
    watershedEx(image, marks);
#else
    watershedOrg(image, marks);
#endif // _MASK	

    Mat watershedImage(marks.rows, marks.cols, CV_8UC3);
    int maxClass = 0, zero = 0;
    for (int i = 0; i < marks.rows; i++){
        for (int j = 0; j < marks.cols; j++){
            int index = marks.at<int>(i, j);
            if (index == -1)
                watershedImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);//watershed line
            else
                watershedImage.at<Vec3b>(i, j) = colorTab[index % 600];//masks with different colors for single region
        }
    }
    imshow("watershed", watershedImage);
    waitKey(0);

    Mat convert_img, wshed;
    image.convertTo(convert_img, CV_8UC3);
    addWeighted(convert_img, 0.7, watershedImage, 0.3, 0, wshed);
    imshow("AddWeighted Image", wshed);
    waitKey();
}
