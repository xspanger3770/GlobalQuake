package globalquake.ui.client;

import globalquake.main.Main;
import globalquake.ui.GQFrame;

import javax.swing.*;
import java.awt.*;

public class ServerSelectionFrame extends GQFrame {

    private JTextField addressField;
    private JTextField portField;

    public ServerSelectionFrame() {
        setTitle(Main.fullName);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setPreferredSize(new Dimension(600,400));

        add(createServerSelectionPanel());

        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private JPanel createServerSelectionPanel() {
        JPanel panel = new JPanel();

        var grid=  new GridLayout(2,1);
        grid.setVgap(5);
        JPanel addressPanel = new JPanel(grid);
        addressPanel.setBorder(BorderFactory.createTitledBorder("Server address"));

        JPanel ipPanel = new JPanel();
        ipPanel.setLayout(new BoxLayout(ipPanel, BoxLayout.X_AXIS));
        ipPanel.add(new JLabel("IP Address: "));
        ipPanel.add(addressField = new JTextField("0.0.0.0",20));

        addressPanel.add(ipPanel);

        JPanel portPanel = new JPanel();
        portPanel.setLayout(new BoxLayout(portPanel, BoxLayout.X_AXIS));
        portPanel.add(new JLabel("Port: "));
        portPanel.add(portField = new JTextField("12345",20));

        addressPanel.add(portPanel);

        panel.add(addressPanel);

        return panel;
    }

    public static void main(String[] args) {
        new ServerSelectionFrame();
    }

}
