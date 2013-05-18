package graph;

import java.util.ArrayList;
import java.util.List;

import dataHandling.MDSMatrix;
import Share.Comm;
import Share.log;

import weka.core.Instance;
import weka.core.Instances;

public class graphHelper {

	/**
	 * Init a graph based on the given instances.
	 * @param graph
	 * @param instances
	 */
	public static void init(Graph graph, Instances instances)
	{
		int docId = 0;
		int numAttr = instances.numAttributes() - 1;  // remvoe the class attribute
		graph.numAttribute = numAttr;
		
		// build nodes
		for(Instance instance : instances)
		{
			Node node = new Node(docId++);
			node.occurrence = new ArrayList<Integer>();
			for(int i = 0; i < numAttr; i++)
			{
				if (instance.value(i) > 0)
				{
					node.occurrence.add(i); // store the attribute id.
				}
			}
			graph.nodes.add(node);
		}

		log.print("Node size is " + graph.nodes.size());
		
		// build edges
		for(int i = 0; i < graph.nodes.size(); i++)
		{
			Node node1 = graph.nodes.get(i);
			for(int j = i+1; j < graph.nodes.size(); j++)
			{
				Node node2 = graph.nodes.get(j);
				int sharedWords = nodeHelper.sharedWords(node1, node2);
				if (sharedWords > Comm.MIN_SharedWord)
				{
					Edge edge = new Edge(node1, node2, sharedWords);
					node1.neighbors.add(node2);
					node2.neighbors.add(node1);
					graph.edges.add(edge);
				}
			}
		}
		
		log.print("Edge size is " + graph.edges.size());
	}
	
	
	/**
	 * Construct the proximity matrix with the given graph.
	 * The proximity is the shortest distance between two nodes / documents.
	 * @param graph
	 */
	public static MDSMatrix buildMatrix(Graph graph)
	{
		int numLine = graph.nodeSize();
		int numCol = graph.attrSize();
		MDSMatrix mat = new MDSMatrix(numLine, numLine);
		int idx = 0;
		for(Node node : graph.nodes)
		{
			double[] distances = nodeHelper.SetShortestDistance(node, graph);
			if (idx % 100 == 0)
			{
				log.print("The " + idx + "th node are set shortest distance.");
			}
			idx++;
		    mat.setLine(distances, node.docId);
		}
		
		return mat;
	}
	
	
	
}
