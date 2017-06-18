package convnet;

import java.util.List;

/**
 * This is an instance of a Convolutional Neural Network built from scratch.
 *    It will feature all the parts of a typical convolutional neural network:
 *       1. Convolutional Layer
 *       2. Pooling Layer
 *       3. Fully Connected Layer
 * 
 * Below is the most general structure of a convolutional neural network:
 * 
 *    Input -> (Conv * i -> Pool * j) * k -> FC * l
 *    
 *    where:
 *       Input is the raw input (3D array of ints)
 *       Conv is one internal convolutional layer
 *       Pool is one pooling layer
 *       FC is one fully-connected layer
 *       k is the number of outer convolutional layers
 *       i is the number of inner convolutional layers
 *       j is the number of inner pooling layers
 * 
 * This structure allows for great complexity and I use some language that
 *    isn't quite typical of what many people may use to describe a
 *    convolutional neural network.
 *    1. Outer convolutional layers:
 *       -  This is the number of combined inner convolutional and pooling
 *          layers.
 *       -  For each outer layer, there may be a number of inner layers
 *          (where 0 <= i,j <= 1), and the number of inner layers is
 *          independent of each outer layer
 *    2. Inner convolutional layers:
 *       -  The typical "convolutional layer" one would think of
 *       -  Contains a certain number of feature maps
 *    3. Inner pooling layers:
 *       -  The typical "pooling layer" one would think of
 *       -  Has the same number of feature maps as the previous inner layer
 *          (whether it was a convolutional or pooling layer)
 * 
 * @author RyanPachauri
 * @version 12/4/16
 */
public class ConvNet {
   
   /*
    * filter/kernel/weights that impact each convolutional layer
    * Each featuer map is affected by a 2D array of weights in addition to one
    *    bias.
    * Here are what each dimension means:
    *    1) The layer between each convolutional layer
    *       (including the input layer)
    *       -> e.g. The first layer (from inputs to convolutional) is
    *          represented by convolutionalWeights[0]
    *    2) The number of feature maps
    *       (i.e. the depth of the output convolutional layer)
    *    3) The number of input layers
    *       - Note that this IS flipped
    *       - Output is a bigger priority than input
    *    4) The rows of the weights
    *    5) The columns of the weights
    *       - The number of rows must match the number of columns
    * 
    * For the last layer of convolutional weights, we will be mapping to a
    *    fully-connected layer.
    * Consider the length of the fully-connected layer
    *    to be the number of feature maps (2) of the fully-connected layer.
    * 
    * If there are k convolutional layers (excluding the input layer),
    *    convolutionalWeights.length will be equal to k + 1.
    */
   private double[][][][][] convolutionalWeights;
   /*
    * Here are what each dimension means:
    *    1) The layer between each convolutional layer
    *       (including the input layer)
    *       -> e.g. The first layer (from inputs to convolutional) is
    *          represented by convolutionalWeights[0]
    *    2) The number of feature maps
    *       (i.e. the depth of the output convolutional layer)
    * We don't need the number of input layers because the bias is only
    *    for determining the activation.
    * We also don't need rows or columns because we only need one number.
    */
   private double[][] convolutionalBiases;
   /* These serve the purpose of providing structure to the network
    * The length of the arrays below is k.
    *    i.e. The length of the outer convolutional layers.
    * The length of the second dimension of each of the arrays may vary (>= 0)
    *    -  The lengths are represented as i (inner convolutional layers) or
    *       j (inner pooling layers)
    * The length of the third dimensions is exactly 2
    *    a. For the convolutional parameters,
    *       i. The first element is the striding parameter
    *       ii.The second element is the padding parameter
    *    b. For the pooling parameters,
    *       i. The first element is the spatial extent parameter
    *       ii.The second element is the striding parameter
    * The values represented are the parameters necessary for the inner layers
    */
   private final int[][] convolutionalParams;
   private final int[][] poolingParams;
   /*
    * weights connecting each 1-dimensional fully-connected layer
    * Here are what each dimension means:
    *    1) Layer of weights between fully-connected layers
    *    2) Length of output fully connected layer
    *    3) Length of input fully connected layer
    * If L represents the number of fully-connected layers, then
    *    outputWeights.length = L - 1
    */
   private double[][][] outputWeights;
   /*
    * Here are what each dimension means:
    *    1) Layer of biases between fully-connected layers
    *    2) Length of output fully connected layer
    *       -  There is a bias for each node (not each layer)
    */
   private double[][] outputBiases;
   /**
    * Use this variable to change the amount we change the weights during
    *    backpropagation.
    */
   public double lamda;
   
   /**
    * @Precondition: 1. convDimensions is not null
    *                2. poolDimensions is not null
    *                3. convDimensions has the same size as poolDimensions
    *                -  If any of the above are false, throws an
    *                      IllegalArgumentException
    */
   public ConvNet(List<int[]> convDimensions, List<int[]> poolDimensions) {
      if (convDimensions == null || poolDimensions == null ||
            convDimensions.size() != poolDimensions.size()) {
         throw new IllegalArgumentException();
      }
      this.convolutionalParams = convDimensions.toArray(new int[0][]);
      this.poolingParams = poolDimensions.toArray(new int[0][]);
   }
   
   /**
    * 
    * @param myConvWeights
    * @param myConvBiases
    * @param convDimensions
    * @param poolDimensions
    * @param myOutputWeights
    * @param myOutputBiases
    */
   public ConvNet(double[][][][][] myConvWeights, double[][] myConvBiases,
         List<int[]> convDimensions, List<int[]> poolDimensions,
         double[][][] myOutputWeights, double[][] myOutputBiases) {
      this(convDimensions, poolDimensions);
      this.convolutionalWeights = myConvWeights;
      this.convolutionalBiases = myConvBiases;
      this.outputWeights = myOutputWeights;
      this.outputBiases = myOutputBiases;
   }

   /**
    * More efficient use of the getConvolution method
    *    (look at getConvolution for a more descriptive comment)
    * @param input   3D array of doubles
    * @param index index of the layer we want the convolutional layer
    * @return  double[][][] convolutional layer at the given index
    */
   private double[][][] getConvolutionAtLayer(double[][][] input, int index) {
      double[][][][] weights = this.convolutionalWeights[index];
      double[] biases = this.convolutionalBiases[index];
      int stride = this.convolutionalParams[index][0];
      int padding = this.convolutionalParams[index][1];
      return this.getConvolution(input, weights, biases, stride, padding);
   }
   
   /**
    * @Preconditions:
    *    1: convolutionalWeights have been set
    *    2: convolutionalBiases have been set
    * @param input   3D array of doubles used in calculating the convolutional
    *                   layer
    * @param weights 4D layer of weights between convolutional layers
    * @param biases  1D array of biases that affect each feature map
    * @param stride  the amount we shift the filter (> 0)
    * @param padding the amount we pad the input's rows and columns
    * @return  double[][][] representing the convolutional layer that is
    *             created with these parameters
    *          Here are how the dimensions are broken down (starting from
    *             outer/most left dimension):
    *             1. Number of feature maps
    *             2. Rows
    *             3. Cols
    */
   private double[][][] getConvolution(double[][][] input, double[][][][] weights,
         double[] biases, int stride, int padding) {
      int numFeatureMaps = weights.length;
      //filter size is constant within weight layers
      int filterRow = weights[0][0].length;
      int rows = (input[0].length - filterRow + 2*padding)/stride + 1;
      int filterCol = weights[0][0].length;
      int cols = (input[0][0].length - filterCol + 2*padding)/stride + 1;
      double[][][] activations = new double[numFeatureMaps][rows][cols];
      input = this.getPaddedInputs(input, padding);
      for (int depth = 0; depth < activations.length; depth++) {
         for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
               double theta = biases[depth];
               theta += this.getConvolutionalTheta(input, row * stride, col *
                     stride, weights[depth],
                     filterRow, filterCol);
               activations[depth][row][col] = this.f(theta);
            }
         }
      }
      return activations;
   }
   
   /**
    * 
    * @param input      3D array of doubles used in calculating theta
    * @param rowStart   starting row of inputs we are concerned with
    * @param colStart   starting col "                             "
    * @param weightLayer   layer of weights between convolutional layers
    * @param featureMap feature map within weightLayer we are calculating for
    * @param filterSize size of the filter of weights (square matrix)
    * @return  double that is ∑k∑r∑c akrc * wrc
    *             where:
    *                a is an input node
    *                k is the layer of the input
    *                r is the row (shifted for inputs)
    *                c is the col (shifted for inputs)
    *                w is a weight
    */
   private double getConvolutionalTheta(double[][][] input, int rowStart,
         int colStart, double[][][] weights, int filterRow, int filterCol) {
      double sum = 0.0;
      for (int depth = 0; depth < input.length; depth++) {
         for (int row = 0; row < filterRow; row++) {
            for (int col = 0; col < filterCol; col++) {
               sum += input[depth][row + rowStart][col + colStart] *
                     weights[depth][row][col];
            }
         }
      }
      return sum;
   }
   
   /**
    * Padding is where you take an array and add zeroes to the borders
    *    This may be useful when creating a convolutional layer so we don't
    *    lose the information at the borders already.
    * 
    * @param input   A 3D array of doubles that we will pad
    * @param padding The amount we will pad to the array on the rows and cols
    * @return
    */
   private double[][][] getPaddedInputs(double[][][] input, int padding) {
      int rows = input[0].length + 2 * padding;
      int cols = input[0][0].length + 2 * padding;
      double[][][] newInputs = new double[input.length][rows][cols];
      for (int depth = 0; depth < input.length; depth++) {
         for (int row = 0; row < input[0].length; row++) {
            for (int col = 0; col < input[0][0].length; col++) {
               newInputs[depth][row + padding][col + padding] =
                     input[depth][row][col];
            }
         }
      }
      return newInputs;
   }
   
   /**
    * More efficient use of the getPoolings method
    *    (look at getPoolings for a more descriptive comment)
    * @param input   3D array of doubles
    * @param index index of the layer we want the convolutional layer
    * @return  double[][][] pooling layer at the given index
    */
   private double[][][] getPoolingsAtLayer(double[][][] input, int index) {
      int spatialExtent = this.poolingParams[index][0];
      int stride = this.poolingParams[index][1];
      return this.getPoolings(input, spatialExtent, stride);
   }
   
   /**
    * 
    * 
    * @param input   a 3D array of doubles
    * @param spatialExtent side length of a square matrix (the pool)
    * @param stride  the amount we shift the rows and cols
    * @param indices a null array that is declared to be
    *       a 4D array of doubles that match the dimensions of the
    *          output; used to keep track of the index of the input where the
    *          output came from
    *       gets around the problem of only being able to return one object
    * @return  A condensed version of input
    */
   private double[][][] getPoolings(double[][][] input, int spatialExtent,
         int stride) {
      int rows = (input[0].length - spatialExtent)/stride + 1;
      int cols = (input[0][0].length - spatialExtent)/stride + 1;
      double[][][] poolings = new double[input.length][rows][cols];
      for (int depth = 0; depth < input.length; depth++) {
         for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
               poolings[depth][row][col] = this.getMaxOfPool(input[depth],
                     row * stride, col * stride, spatialExtent);
            }
         }
      }
      return poolings;
   }
   
   /**
    * This is a method of pooling called max pooling.
    *    Within one square matrix of input, returns the largest double
    *    We want to do this because that means we get the most effective node
    *       from the input
    * @param input   2D array of doubles
    * @param rowStart   starting row within the input
    * @param colStart   starting col within the input
    * @param spatialExtent side length of the square matrix (a subsection of
    *                      the input array)
    * @return  the largest double
    */
   private double getMaxOfPool(double[][] input, int rowStart, int colStart,
         int spatialExtent) {
      double max = input[rowStart][colStart];
      for (int row = rowStart; row < rowStart + spatialExtent; row++) {
         for (int col = colStart; col < colStart + spatialExtent; col++) {
            max = Math.max(max, input[row][col]);
         }
      }
      return max;
   }
   
   /**
    * Connecting the last layer of 3D arrays to the first fully-connected
    *    layer (which are all 1D)
    * @param input   3D array of doubles
    * @return  1D array of doubles
    */
   private double[] getFullyConnecteds(double[][][] input,
         double[][][][] weights, double[] biases) {
      double[] output = new double[weights.length];
      for (int i = 0; i < output.length; i++) {
         double theta = biases[i];
         theta += this.getConvolutionalTheta(input, 0, 0,
               weights[i], input[0].length, input[0][0].length);
         output[i] = this.f(theta);
      }
      return output;
   }
   
   /**
    * Connecting all of the output layers except for the last one
    * 
    * @Precondition:
    *    1: input.length equals this.outputWeights[weightLayer][0].length
    *    2: biases.length = weights.length
    * Checking for the above and throwing errors is computationally expensive
    *    so any client of this code must be careful when passing in parameters.
    * 
    * @param input   1D array of doubles
    * @param weightLayer   2D array of doubles where:
    *          1) First dimension is the length of the number of outputs
    *          2) Second dimension is the length of the number of inputs
    * @param biases  1D array of doubles
    * @return  1D array of activated nodes using inputs, weights, and biases
    */
   private double[] getFullyConnecteds(double[] input, double[][] weights,
         double[] biases) {
      double[] output = new double[weights.length];
      for (int l = 0; l < output.length; l++) {
         double theta = biases[l];
         for (int i = 0; i < input.length; i++) {
            theta += input[i] * weights[l][i];
         }
         output[l] = this.f(theta);
      }
      return output;
   }
   
   /**
    * This is our activation function. We use this in feeding forward for our
    *    neural network.
    * 
    * ***If you change this, then you must change fPrime as well ***
    * 
    * f(x) = 1 / (1 + e^ (-x))
    * 
    * @param x    the double we are putting into the function
    * @return     the value when x is put into the function
    */
   private double f(double x) {
      return (1 / (1 + Math.exp(-x)));
   }
   
   /**
    * This is the derivative of the activation function.
    * 
    * The derivative of f(x) = 1 / (1 + e^ (-x))
    *    is simply f'(x) = f(x) * (1 - f(x))
    * 
    * @param x    the double we are putting into the derivative of the
    *                activation function.
    *                We are assuming that x is already the result of f(x).
    * @return     the value when x is put into the derivative of the activation
    *                function
    */
   private double sigmoidPrime(double x) {
      return x * (1 - x);
   }
   
   /**
    * All of the following parameters are 1D arrays of doubles of the same
    *    length = n
    * 
    * @param output  actual output we compare to expected
    * @param expected   what we expect output should be
    * @param biases  biases for the output
    * @param psi     each value is represented by the following equation:
    *       psii = (yi - oi) * f'(thetai)
    * @return  double the error represented by the following equation:
    * 
    * E = 1/2 * ∑ (yi - oi)^2
    */
   private double getError(double[] output, int[] expected, double[] biases,
         double[] psi) {
      int n = output.length;
      double error = 0.0;
      for (int i = 0; i < n; i++) {
         double delta = expected[i] - output[i];
         psi[i] = delta * this.sigmoidPrime(output[i]);
         error += delta * delta;
         biases[i] += this.lamda * psi[i];
      }
      return error/2;
   }
   
   /**
    * Remember that l represents the index of the fully-connected layers
    * 
    * l                 l + 1             ...         L
    * 
    * a1                  a1                          o1
    * a2                  a2                          o2
    * .                   .                           .
    * .                   .                           oN
    * .                   an
    * am
    * 
    * In the above diagram, the l column represents the "inputs" and the
    *    (l + 1) column represents the next layer.
    *    For this method, we will be using j to represent l and i to represent
    *    (l + 1).
    * The L column represents the output of the network (calculating psiis
    *    is different for those nodes).
    * 
    * @param input   1D array of fully-connected nodes (l)
    * @param weights 2D array of weights connecting input and output
    * @param biases  biases for the inputs
    * @param psiis   ∂C/∂z(l + 1) of the next layer (the derivative of the Cost
    *                   function with respect to the output nodes)
    * @return psijs  ∂C/∂z(l) of the inputs (the derivate of the Cost function
    *                   with respect to the input nodes)
    * 
    * Calculating psijs:
    * 1. Omegaj = Sum the product of (psiis * wji)
    *    -  Where wji is any weight connected to the input node
    * 2. Delta wji = lambda * aj * psii
    *    -  Where aj = input[j] = f(thetaj)
    * 3. Psij = omegaj * f'(thetaj)
    *    -  Where f'(thetaj) = inpu
    * 
    * Note that in this method, we will also be improving the weights.
    *    It is imperative that we calculate the omegajs before improving the
    *    weights.
    *    Since the omegajs are dependent on the OLD wjis, calculating them
    *    after the wieghts have improved could lead to skewed results.
    */
   private double[] getFullyConnectedPsis(double[] input, double[][] weights,
         double[] biases, double[] psiis) {
      int m = input.length;
      int n = psiis.length;
      double[] psijs = new double[m];
      for (int j = 0; j < m; j++) {
         double omegaj = 0.0;
         double factor = this.lamda * input[j];
         for (int i = 0; i < n; i++) {
            omegaj += psiis[i] * weights[i][j];
            weights[i][j] += factor * psiis[i];
         }
         psijs[j] = omegaj * this.sigmoidPrime(input[j]);
         biases[j] += this.lamda * psijs[j];
      }
      return psijs;
   }
   
   /**
    * Getter for convolutional parameters
    * @return  3D array of ints
    */
   public int[][] getConvolutionalParameters() {
      return this.convolutionalParams;
   }
   
   /**
    * Getter for pooling parameters
    * @return  3D array of ints
    */
   public int[][] getPoolingParameters() {
      return this.poolingParams;
   }
   
   /**
    * Setter for convolutional weights
    * @param weights 4D array of doubles
    */
   public void setConvolutionalWeights(double[][][][][] weights) {
      this.convolutionalWeights = weights;
   }
   
   /**
    * Setter for convolutional biases
    * @param biases  2D array of doubles
    */
   public void setConvolutionalBiases(double[][] biases) {
      this.convolutionalBiases = biases;
   }
   
   /**
    * Setter for fully-connected weights
    * @param weights 3D array of doubles
    */
   public void setOutputWeights(double[][][] weights) {
      this.outputWeights = weights;
   }
   
   /**
    * Setter for fully-connected biases
    * @param biases  2D array of doubles
    */
   public void setOutputBiases(double[][] biases) {
      this.outputBiases = biases;
   }
   
   /**
    * @Precondition: 1. Weights and biases are set to non-zero values
    *                2. There is at least one outer convolutional layer
    * @param input   a 3D array of double
    *          This will preferably be raw input.
    * @param expectedOutput   a 1D array of ints
    *          The typical format of this array is that the values at all
    *             indices are set to 0 and only one index has a value of 1.
    *          While this is not necessary, it is necessary that the range of
    *             values at all indices is from 0 to 1 (inclusive).
    * @return  double representing the standard error
    * 
    * Recall the error function:
    * 
    *    E = 1/n ∑i (Ti - Fi)^2
    *    
    *    where
    *    Fi is the Output Activation (what we got)
    *    Ti is the Target Output Activation (what we want)
    *    i is the target node index
    *    n is the number of target nodes
    *    E is the error function (standard error)
    */
   public double train(double[][][] input, int[] expectedOutput) {
      //1. forward the first convolutional layer
      //2. for loop to forward the other layers
      int numConvLayers = this.convolutionalParams.length;
      double[][][][] convLayers = new double[numConvLayers][][][];
      double[][][][] poolLayers = new double[numConvLayers][][][];
      for (int k = 0; k < numConvLayers; k++) {
         //have to check whether or not there is an inner convolutional layer
         if (this.convolutionalParams[k] != null) {
            convLayers[k] = this.getConvolutionAtLayer(input, k);
            input = convLayers[k];
         }
         if (this.poolingParams[k] != null) {
            poolLayers[k] = this.getPoolingsAtLayer(input, k);
            input = poolLayers[k];
         }
      }
      //3. forward the last convolutional layer to the first fully-connected
      //   layer
      int numFullyConnectedLayers = this.outputWeights.length + 1;
      double[][] fullyConnecteds = new double[numFullyConnectedLayers][];
      fullyConnecteds[0] = this.getFullyConnecteds(input,
            this.convolutionalWeights[numConvLayers],
            this.convolutionalBiases[numConvLayers]);
      //4. forward the rest of the full-connected layers (including the output)
      //s represents the index of the last fully-connected layer (output)
      int s = numFullyConnectedLayers - 1;
      for (int l = 0; l < s; l++) {
         fullyConnecteds[l+1] = this.getFullyConnecteds(fullyConnecteds[l],
               this.outputWeights[l], this.outputBiases[l]);
      }
      
      //5. get the output (and deltas) as well as the error
      double[] psi = new double[expectedOutput.length];
      double error = this.getError(fullyConnecteds[s], expectedOutput, psi);
      //6. start backpropagating the fully-connected layers TODO
      //7. backpropagate the first fully-connected layer to the last convolutional layer
      //8. backpropagate the convolutional layers
      return error;
   }
   
   /**
    * 
    * @param input a 3D array of ints
    *          This will preferably be raw input.
    * @return an array of doubles (the output when the input is evaluated
    *          across this network)
    */
   public double[] evaluateNetwork(int[][][] input) {
      //1. forward the first convolutional layer
      //2. for loop to forward the other layers
      //3. forward the last convolutional layer to the first fully-connected layer
      //4. forward the rest of the full-connected layers (including the output)
      //5. return the last fully-connected layer (as the output)
      //*Note that 1-3 is exactly the same as the train method (with a minor tweak for 4)
      return null;
   }
}
