package convnet;

import java.util.*;

/**
 * Runner for the class ConvNet
 * @author RyanPachauri
 *
 */
public class ConvNetRunner {

   public static void main(String[] args) {
      System.out.println("Welcome to Ryan Pachauri's Convolutional Neural"
            + " Network!");
      System.out.println("With this program, you can train a neural network to"
            + " recognize images.");
      Scanner console = new Scanner(System.in);
      int procedure = getProcedures(console);
      console.close();
      setWeightsAndBiases(procedure);
      //train(convolutionalNetwork);TODO
   }
   
   /**
    * (1)   Create a new network
    *          Using random filters for each of the weights
    * (2)   Read from an input file
    *          Where all of the weights are already defined
    * @return  an integer corresponding to the procedure for creating a
    *             ConvNet object
    */
   private static int getProcedures(Scanner console) {
      System.out.println("How would you like to proceed?");
      List<String> procedures = new ArrayList<String>();
      procedures.add("Create a new network?");
      procedures.add("Read from an input file?");
      for(int i = 0; i < procedures.size(); i++) {
         System.out.println("\t(" + (i + 1) + ")\t" + procedures.get(i));
      }
      int reply;
      do {
         System.out.println("Please input one of the above numbers");
         reply = console.nextInt();
      } while (reply <= 0 || reply > procedures.size());
      return reply;
   }
   
   /**
    * @Precondition: reply has to be one of the numbers described in the
    *                   getProcedures method
    * @param reply   one of the above numbers
    * @return  a ConvNet object that is created according to the procedure
    *             defined by reply
    */
   private static void setWeightsAndBiases(int reply) {
      switch (reply) {
      case 1:
         createNetworkFromScratch();
      case 2:
         createNetworkFromFile();
      }
      //TODO
   }
   
   private static void createNetworkFromScratch() {
      //TODO
   }
   
   public static ConvNet createNetwork(List<Integer> fullyConnectedWeights) {
      return null;//TODO
   }
   
   private static void createNetworkFromFile() {
      //TODO
   }
   
   private static int[][] getInputs(Scanner input) {
      return null;//TODO
   }
   
   private static int[] getExpectedOutput(Scanner input) {
      return null;//TODO
   }
   
   private static void train(ConvNet cnn) {
      //TODO
   }
}
