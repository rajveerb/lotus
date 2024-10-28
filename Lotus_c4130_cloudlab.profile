"""Single node with dataset"""

#
# NOTE: This code was machine converted. An actual human would not
#       write code like this!
#

# Import the Portal object.
import geni.portal as portal
# Import the ProtoGENI library.
import geni.rspec.pg as rspec
# Import the Emulab specific extensions.
import geni.rspec.emulab as emulab

pc = portal.Context();


# Create a portal object,
pc.defineParameter("hardware_type", "Your hardware node",
                   portal.ParameterType.STRING,
                   "c4130")
pc.defineParameter("dataset", "Your dataset URN",
                   portal.ParameterType.STRING,
                   "urn:publicid:IDN+emulab.net:portalprofiles+ltdataset+DemoDataset")

                   
# Create a Request object to start building the RSpec.
request = pc.makeRequestRSpec()

# Get parameter values
params = pc.bindParameters()

# Node c4130-node
node_type = request.RawPC(params.hardware_type +'-node')
node_type.hardware_type = params.hardware_type
node_type.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU20-64-STD";

iface = node_type.addInterface()

# The remote file system is represented by special node.
fsnode = request.RemoteBlockstore("fsnode", "/mydata")
# This URN is displayed in the web interfaace for your dataset.
fsnode.dataset = params.dataset

# Now we add the link between the node and the special node
fslink = request.Link("fslink")
fslink.addInterface(iface)
fslink.addInterface(fsnode.interface)

# Special attributes for this link that we must use.
fslink.best_effort = True
fslink.vlan_tagging = True


# Print the generated rspec
pc.printRequestRSpec(request)