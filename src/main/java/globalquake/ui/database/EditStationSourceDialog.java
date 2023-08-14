package globalquake.ui.database;

import globalquake.database.StationDatabaseManager;
import globalquake.database.StationSource;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class EditStationSourceDialog extends JDialog {
    private final StationDatabaseManager databaseManager;
    private final StationSource stationSource;
    private final JTextField nameField;
    private final JTextField urlField;

    public EditStationSourceDialog(Window parent, StationDatabaseManager databaseManager, StationSource stationSource) {
        super(parent);
        setModal(true);

        this.databaseManager = databaseManager;
        this.stationSource = stationSource;
        setLayout(new BorderLayout());

        setTitle("Edit Station Source");
        setSize(320, 180);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        setLocationRelativeTo(parent);

        JPanel fieldsPanel = new JPanel();
        LayoutManager gridLayout = new BoxLayout(fieldsPanel, BoxLayout.Y_AXIS);
        fieldsPanel.setLayout(gridLayout);
        fieldsPanel.setBorder(new EmptyBorder(10,10,10,10));

        nameField = new JTextField(stationSource==null ? "" : stationSource.getName(), 40);
        urlField = new JTextField(stationSource==null ? "" : stationSource.getUrl(), 40);
        JButton saveButton = new JButton("Save");
        saveButton.addActionListener(e -> saveChanges());
        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(actionEvent -> EditStationSourceDialog.this.dispose());

        JPanel buttonsPanel = new JPanel();

        fieldsPanel.add(new JLabel("Name:"));
        fieldsPanel.add(nameField);
        fieldsPanel.add(new JLabel("URL:"));
        fieldsPanel.add(urlField);
        buttonsPanel.add(cancelButton);
        buttonsPanel.add(saveButton);

        add(fieldsPanel, BorderLayout.CENTER);
        add(buttonsPanel, BorderLayout.SOUTH);

        setResizable(false);

        pack();
        setVisible(true);
    }

    private void saveChanges() {
        StationSource newStationSource = new StationSource(nameField.getText(), urlField.getText());
        databaseManager.getStationDatabase().getDatabaseWriteLock().lock();
        try{
            databaseManager.getStationDatabase().getStationSources().remove(stationSource);
            databaseManager.getStationDatabase().getStationSources().add(newStationSource);
        }finally {
            databaseManager.getStationDatabase().getDatabaseWriteLock().unlock();
        }

        databaseManager.fireUpdateEvent();

        this.dispose();
    }
}
